import type {
	LanguageModelV4,
	LanguageModelV4CallOptions,
	LanguageModelV4Content,
	LanguageModelV4StreamPart,
	SharedV4Warning,
} from "@ai-sdk/provider";
import {
	convertAsyncIteratorToReadableStream,
	createNullLanguageModelUsage,
	parseProviderOptions,
} from "@ai-sdk/provider-utils";
import { answersFor, rebuildItems, sliceHistory } from "./history.ts";
import type {
	CodexDynamicToolSpec,
	CodexReasoningEffort,
	CodexResponseItem,
	CodexThreadResponse,
	CodexThreadStartParams,
	CodexUserInput,
} from "./host/app-server.ts";
import type { CodexHost } from "./host/protocol.ts";
import {
	type CodexModelId,
	CodexOptions,
	type CodexProviderSettings,
} from "./options.ts";
import {
	defaultTranscript,
	findCursor,
	getDangling,
	getSystem,
	getTail,
	METADATA_KEY,
	toInput,
} from "./prompt.ts";
import { Session, type SessionRegistry, type StepContext } from "./session.ts";
import { REQUEST_USER_INPUT, TOOL_ID_PREFIX } from "./tools.ts";

const PROVIDER = "codex";

type Reasoning = NonNullable<LanguageModelV4CallOptions["reasoning"]>;

const efforts: Record<Reasoning, CodexReasoningEffort | undefined> = {
	"provider-default": undefined,
	none: "none",
	minimal: "low",
	low: "low",
	medium: "medium",
	high: "high",
	xhigh: "xhigh",
	max: "max",
};

const unsupportedSettings = [
	"maxOutputTokens",
	"temperature",
	"topP",
	"topK",
	"stopSequences",
	"presencePenalty",
	"frequencyPenalty",
	"seed",
] as const satisfies (keyof LanguageModelV4CallOptions)[];

/** How long the user's MCP servers, read to turn them off, are trusted. */
const USER_SERVERS_MS = 60_000;

/** A short hash of a string, to compare settings by. */
const hash = (value: string) => {
	let h = 0x811c9dc5;
	for (let i = 0; i < value.length; i++) {
		h ^= value.charCodeAt(i);
		h = Math.imul(h, 0x01000193);
	}
	return (h >>> 0).toString(16).padStart(8, "0");
};

/** JSON with objects' keys in order, so equal settings read the same. */
const stableJson = (value: unknown): string =>
	JSON.stringify(value, (_, inner) =>
		inner && typeof inner === "object" && !Array.isArray(inner)
			? Object.fromEntries(
					Object.entries(inner).sort(([a], [b]) => (a < b ? -1 : 1)),
				)
			: inner,
	);

const text = (value: string): CodexUserInput => ({
	type: "text",
	text: value,
	text_elements: [],
});

/** The config that turns Codex's tools on and off, by the tools the call has. */
function toolConfig(
	builtins: "codex" | Set<string>,
	asks: boolean,
): Record<string, unknown> {
	const config: Record<string, unknown> = {
		// Asking needs a caller to answer.
		"tools.experimental_request_user_input.enabled": asks,
		"features.default_mode_request_user_input": asks,
		...(asks && { suppress_unstable_features_warning: true }),
	};
	if (builtins === "codex") return config;
	const on = (name: string) => builtins.has(name);
	return {
		...config,
		"features.shell_tool": on("exec_command"),
		"features.unified_exec": on("exec_command"),
		"features.shell_snapshot": on("exec_command"),
		web_search: on("web_search") ? "live" : "disabled",
		"features.standalone_web_search": false,
		"features.view_image": on("view_image"),
		"features.image_generation": on("image_generation"),
		"tools.update_plan.enabled": on("update_plan"),
		"features.multi_agent": on("spawn_agent"),
		"features.multi_agent_v2.disable_native_tools": !on("spawn_agent"),
		"features.request_permissions_tool": false,
		"features.tool_suggest": false,
		"features.goals": false,
		"features.sleep_tool": false,
		"features.current_time_reminder": false,
		"features.token_budget": false,
	};
}

/** What a call to the model sets up, from its options. */
async function prepare(
	modelId: string,
	options: LanguageModelV4CallOptions,
	settings: CodexProviderSettings,
	modelDefaults: CodexOptions | undefined,
	userServers: (cwd: string | undefined) => Promise<string[]>,
) {
	const warnings: SharedV4Warning[] = [];
	const own = await parseProviderOptions({
		provider: METADATA_KEY,
		providerOptions: options.providerOptions,
		schema: CodexOptions,
	});
	const opts: CodexOptions = {
		...settings.defaults,
		...modelDefaults,
		...own,
	};

	for (const setting of unsupportedSettings)
		if (options[setting] !== undefined)
			warnings.push({ type: "unsupported", feature: setting });

	// The call's tools: its own, as Codex's dynamic tools, and Codex's it declared.
	const declared = new Map<string, string>();
	let questionKey: string | undefined;
	const tools: CodexDynamicToolSpec[] = [];
	for (const tool of options.tools ?? []) {
		if (tool.type === "function")
			tools.push({
				type: "function",
				name: tool.name,
				description: tool.description ?? "",
				inputSchema: tool.inputSchema as never,
			});
		else if (tool.id.startsWith(TOOL_ID_PREFIX)) {
			const name = tool.id.slice(TOOL_ID_PREFIX.length);
			if (name === REQUEST_USER_INPUT) questionKey = tool.name;
			else declared.set(name, tool.name);
		} else
			warnings.push({
				type: "unsupported",
				feature: `provider tool ${tool.id}`,
			});
	}
	const choice = options.toolChoice?.type;
	if (choice && choice !== "auto")
		warnings.push({
			type: "unsupported",
			feature: `toolChoice: ${choice}`,
			details: "Codex picks its own tools",
		});

	const builtins =
		opts.builtinTools === "codex"
			? ("codex" as const)
			: new Set([...(opts.builtinTools ?? []), ...declared.keys()]);
	const applyPatch = builtins === "codex" || builtins.has("apply_patch");

	let outputSchema: unknown;
	if (options.responseFormat?.type === "json") {
		if (options.responseFormat.schema)
			outputSchema = options.responseFormat.schema;
		else
			warnings.push({
				type: "unsupported",
				feature: "responseFormat without a schema",
			});
	}

	const config: Record<string, unknown> = {
		"features.code_mode.default_exec_yield_time_ms":
			opts.execYieldMs ?? 3_600_000,
		...toolConfig(builtins, !!questionKey),
	};
	if (!opts.userConfig) {
		Object.assign(config, {
			"features.apps": false,
			"features.plugins": false,
			"features.memories": false,
			"features.hooks": false,
			"skills.include_instructions": false,
			"cloud.skills.enabled": false,
			project_doc_max_bytes: 0,
		});
		const servers = await userServers(opts.cwd);
		if (servers.length)
			config.mcp_servers = Object.fromEntries(
				servers.map((name) => [name, { enabled: false }]),
			);
	}
	if (opts.mcpServers)
		config.mcp_servers = {
			...(config.mcp_servers as object | undefined),
			...opts.mcpServers,
		};
	Object.assign(config, opts.config);

	const system = getSystem(options.prompt);
	const model = modelId === "default" ? undefined : modelId;
	const sandbox =
		opts.sandbox ??
		(!applyPatch && !opts.permissions ? ("read-only" as const) : undefined);
	const thread: CodexThreadStartParams = {
		model,
		cwd: opts.cwd,
		...(opts.cwd &&
			opts.additionalDirectories?.length && {
				runtimeWorkspaceRoots: [opts.cwd, ...opts.additionalDirectories],
			}),
		approvalPolicy: opts.approvalPolicy,
		sandbox,
		permissions: opts.permissions,
		serviceTier: opts.serviceTier,
		...(opts.systemPromptPreset
			? { developerInstructions: system }
			: { baseInstructions: system ?? "" }),
		config: config as never,
		dynamicTools: tools,
		ephemeral: opts.persistThread === false,
		experimentalRawEvents: true,
		...opts.threadOptions,
	};
	for (const key of Object.keys(thread))
		if (thread[key] === undefined) delete thread[key];

	const transcript = settings.transcript ?? defaultTranscript;
	const ctx: StepContext = {
		declared,
		questionKey,
		tools,
		model,
		turn: {
			model,
			cwd: opts.cwd,
			approvalPolicy: opts.approvalPolicy,
			serviceTier: opts.serviceTier,
			effort: opts.effort ?? efforts[options.reasoning ?? "provider-default"],
			summary: opts.reasoningSummary,
			outputSchema: outputSchema as never,
		},
		fingerprint: hash(
			stableJson({
				baseInstructions: thread.baseInstructions,
				developerInstructions: thread.developerInstructions,
				dynamicTools: thread.dynamicTools,
				config: thread.config,
				ephemeral: thread.ephemeral,
			}),
		),
		applyPatch,
		includeRawChunks: options.includeRawChunks ?? false,
		warnings,
		abortSignal: options.abortSignal,
		toInput: (messages) =>
			toInput(messages, {
				convertFile: settings.convertFile,
				warnings,
				transcript,
			}),
	};
	for (const key of Object.keys(ctx.turn) as (keyof StepContext["turn"])[])
		if (ctx.turn[key] === undefined) delete ctx.turn[key];
	return { opts, thread, ctx, transcript };
}

export function createCodexModel({
	modelId,
	defaults,
	settings,
	getHost,
	registry,
}: {
	modelId: CodexModelId;
	defaults?: CodexOptions;
	settings: CodexProviderSettings;
	getHost: () => Promise<CodexHost>;
	registry: SessionRegistry;
}): LanguageModelV4 {
	const hooks = registry.hooks(settings.sessionTtlMs ?? 10 * 60_000);

	const servers = new Map<string, { names: string[]; at: number }>();
	/** The MCP servers the user's config has, to turn them off. */
	const userServers = async (cwd: string | undefined) => {
		const key = cwd ?? "";
		const cached = servers.get(key);
		if (cached && Date.now() - cached.at < USER_SERVERS_MS) return cached.names;
		const host = await getHost();
		const read = (await host
			.request("config/read", { includeLayers: false, cwd: cwd ?? null })
			.catch(() => undefined)) as { config?: Record<string, any> } | undefined;
		const found =
			read?.config?.mcp_servers ?? read?.config?.additional?.mcp_servers ?? {};
		const names = Object.keys(found).sort();
		servers.set(key, { names, at: Date.now() });
		return names;
	};

	/** Starts a thread, with history to go on from, and its first turn. */
	const startThread = async (
		host: CodexHost,
		prepared: Awaited<ReturnType<typeof prepare>>,
		seed: CodexResponseItem[],
		input: CodexUserInput[],
	) => {
		const { thread } = (await host.request(
			"thread/start",
			prepared.thread,
		)) as CodexThreadResponse;
		const session = new Session(host, thread.id, prepared.ctx, hooks, seed);
		try {
			if (seed.length)
				await host.request("thread/inject_items", {
					threadId: thread.id,
					items: seed,
				});
			await session.startTurn(input.length ? input : [text("Continue.")]);
		} catch (error) {
			await session.close();
			throw error;
		}
		return session;
	};

	/** The thread a call goes on in: continued, rebuilt at its cursor, or new. */
	const getSession = async (
		options: LanguageModelV4CallOptions,
		prepared: Awaited<ReturnType<typeof prepare>>,
	) => {
		const { ctx, transcript } = prepared;
		const host = await getHost();
		registry.attach(host);
		const prompt = options.prompt;

		const cursor = findCursor(prompt);
		if (cursor) {
			const tail = getTail(prompt, cursor.index + 1);
			const live = registry.get(cursor.threadId);
			if (live?.canContinue(cursor.cursor, ctx.fingerprint)) {
				await live.update(ctx);
				await live.deliver(tail);
				return { session: live, route: "continue" as const };
			}
			// A new thread with the old one's history up to the cursor: Codex
			// streams its API calls only to threads it starts.
			const entries = live?.history() ?? (await host.history(cursor.threadId));
			const history = entries && sliceHistory(entries, cursor);
			if (history) {
				const answers = answersFor(getDangling(prompt, cursor), tail);
				const seed = rebuildItems(history, answers, {
					keepReasoning: !cursor.model || cursor.model === ctx.model,
				});
				const input = await ctx.toInput(tail.messages);
				const session = await startThread(host, prepared, seed, input);
				// Its tools or instructions changed: the new thread replaces it.
				if (live?.isAt(cursor.cursor)) await live.close();
				return { session, route: "rebuild" as const };
			}
		}

		// A new thread, told what came before.
		let start = prompt.length;
		while (start > 0 && prompt[start - 1].role === "user") start--;
		while (start > 0 && prompt[start - 1].role === "system") start--;
		const history = transcript(
			prompt.slice(0, start).filter((message) => message.role !== "system"),
		);
		const turn = prompt.slice(start).filter((m) => m.role === "user");
		const input = [
			...(history ? [text(history)] : []),
			...(await ctx.toInput(turn)),
		];
		const session = await startThread(host, prepared, [], input);
		return { session, route: "new" as const };
	};

	return {
		specificationVersion: "v4",
		provider: PROVIDER,
		modelId,
		supportedUrls: { "image/*": [/^https?:\/\//] },

		async doStream(options) {
			const prepared = await prepare(
				modelId,
				options,
				settings,
				defaults,
				userServers,
			);
			const { session, route } = await getSession(options, prepared);
			prepared.ctx.route = route;
			return {
				stream: convertAsyncIteratorToReadableStream(
					session.stream(prepared.ctx),
				),
				request: { body: { route, thread: prepared.thread } },
			};
		},

		async doGenerate(options) {
			const { stream, request } = await this.doStream(options);
			const content: LanguageModelV4Content[] = [];
			const blocks = new Map<string, { text: string }>();
			let warnings: SharedV4Warning[] = [];
			let finish:
				| Extract<LanguageModelV4StreamPart, { type: "finish" }>
				| undefined;
			let response: { id?: string; modelId?: string; timestamp?: Date } = {};
			let error: unknown;
			for await (const part of stream) {
				switch (part.type) {
					case "stream-start":
						warnings = part.warnings;
						break;
					case "response-metadata":
						response = { ...response, ...part };
						break;
					case "text-start":
					case "reasoning-start": {
						const block = {
							type: part.type === "text-start" ? "text" : "reasoning",
							text: "",
						} as LanguageModelV4Content & { text: string };
						blocks.set(part.id, block);
						content.push(block);
						break;
					}
					case "text-delta":
					case "reasoning-delta": {
						const block = blocks.get(part.id);
						if (block) block.text += part.delta;
						break;
					}
					case "text-end":
					case "reasoning-end": {
						const block = blocks.get(part.id) as
							| (LanguageModelV4Content & { providerMetadata?: unknown })
							| undefined;
						if (block && part.providerMetadata)
							block.providerMetadata = part.providerMetadata;
						break;
					}
					case "tool-result":
						if (part.preliminary) break;
						content.push(part);
						break;
					case "tool-call":
					case "tool-approval-request":
					case "custom":
					case "file":
					case "source":
						content.push(part);
						break;
					case "error":
						error ??= part.error;
						break;
					case "finish":
						finish = part;
						break;
				}
			}
			if (error && !content.length) throw error;
			return {
				content,
				finishReason: finish?.finishReason ?? {
					unified: "other",
					raw: undefined,
				},
				usage: finish?.usage ?? createNullLanguageModelUsage(),
				providerMetadata: finish?.providerMetadata,
				warnings,
				request,
				response: {
					id: response.id,
					modelId: response.modelId,
					timestamp: response.timestamp,
				},
			};
		},
	};
}
