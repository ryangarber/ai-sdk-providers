import type {
	JSONObject,
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
import type {
	CursorAgentOptions,
	CursorHost,
	CursorModelInfo,
	CursorModelParameter,
	CursorToolDefinition,
	CursorUserMessage,
} from "./host/protocol.ts";
import {
	type CursorModelId,
	CursorOptions,
	type CursorProviderSettings,
} from "./options.ts";
import {
	defaultTranscript,
	findCursor,
	getSystem,
	getTail,
	METADATA_KEY,
	toMessage,
} from "./prompt.ts";
import { Session, type StepContext } from "./session.ts";
import { TOOL_GROUPS, TOOL_ID_PREFIX } from "./tools.ts";

const PROVIDER = "cursor";

type Reasoning = NonNullable<LanguageModelV4CallOptions["reasoning"]>;

/** The parameters models take effort as. */
const EFFORT_PARAMS = ["effort", "reasoning_effort", "reasoning"];

/** A call's `reasoning`, as the values a model may take it as, best first. */
const efforts: Record<Exclude<Reasoning, "provider-default">, string[]> = {
	none: ["none"],
	minimal: ["minimal", "low"],
	low: ["low"],
	medium: ["medium"],
	high: ["high"],
	xhigh: ["xhigh", "extra-high", "high"],
	max: ["max", "xhigh", "extra-high", "high"],
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

/** The agents a provider has open, by agent id. */
export class SessionRegistry {
	private sessions = new Map<string, Session>();
	get = (id: string) => this.sessions.get(id);
	register = (session: Session) => this.sessions.set(session.agentId, session);
	unregister = (session: Session) => {
		if (this.sessions.get(session.agentId) === session)
			this.sessions.delete(session.agentId);
	};
	all = () => [...this.sessions.values()];
}

/** The model's parameters for a call's effort, and its own. */
export function modelParams(
	model: CursorModelInfo | undefined,
	reasoning: LanguageModelV4CallOptions["reasoning"],
	effort: string | undefined,
	warnings: SharedV4Warning[],
): CursorModelParameter[] {
	if (!effort && (!reasoning || reasoning === "provider-default")) return [];
	const params = model?.parameters ?? [];
	const param = params.find((p) => EFFORT_PARAMS.includes(p.id));
	const values = new Set(param?.values.map((v) => v.value));
	if (effort) {
		if (param) return [{ id: param.id, value: effort }];
		warnings.push({
			type: "unsupported",
			feature: "effort",
			details: `${model?.id ?? "This model"} takes no effort`,
		});
		return [];
	}
	const level = reasoning as Exclude<Reasoning, "provider-default">;
	const value = efforts[level].find((candidate) => values.has(candidate));
	if (param && value) return [{ id: param.id, value }];
	const thinking = params.find((p) => p.id === "thinking");
	if (level === "none" && thinking) return [{ id: "thinking", value: "false" }];
	if (level !== "none" && thinking && !param)
		return [{ id: "thinking", value: "true" }];
	warnings.push({
		type: "unsupported",
		feature: `reasoning: ${level}`,
		details: `${model?.id ?? "This model"} doesn't take it`,
	});
	return [];
}

/** What a call to the model sets up, from its options. */
async function prepare(
	modelId: string,
	options: LanguageModelV4CallOptions,
	settings: CursorProviderSettings,
	modelDefaults: CursorOptions | undefined,
	getHost: () => Promise<CursorHost>,
) {
	const warnings: SharedV4Warning[] = [];
	const own = await parseProviderOptions({
		provider: METADATA_KEY,
		providerOptions: options.providerOptions,
		schema: CursorOptions,
	});
	const opts: CursorOptions = {
		...settings.defaults,
		...modelDefaults,
		...own,
	};

	for (const setting of unsupportedSettings)
		if (options[setting] !== undefined)
			warnings.push({ type: "unsupported", feature: setting });

	// The call's tools: its own, served as custom tools, and Cursor's it declared.
	const declared = new Map<string, string>();
	let tools: CursorToolDefinition[] = [];
	for (const tool of options.tools ?? []) {
		if (tool.type === "function")
			tools.push({
				name: tool.name,
				description: tool.description,
				inputSchema: tool.inputSchema as JSONObject,
			});
		else if (tool.id.startsWith(TOOL_ID_PREFIX))
			declared.set(tool.id.slice(TOOL_ID_PREFIX.length), tool.name);
		else
			warnings.push({
				type: "unsupported",
				feature: `provider tool ${tool.id}`,
			});
	}

	let builtins: string[] | undefined =
		opts.builtinTools === "cursor"
			? undefined
			: [
					...new Set([
						...(opts.builtinTools ?? []),
						...[...declared.keys()].map((name) => TOOL_GROUPS[name] ?? name),
						// The call's tools and MCP servers come through Cursor's MCP.
						...(tools.length || opts.mcpServers ? ["mcp"] : []),
					]),
				];
	const choice = options.toolChoice?.type;
	if (choice === "none") {
		tools = [];
		builtins = [];
	} else if (choice === "required" || choice === "tool")
		warnings.push({
			type: "unsupported",
			feature: `toolChoice: ${choice}`,
			details: "Cursor picks its own tools",
		});

	if (options.responseFormat?.type === "json")
		warnings.push({
			type: "unsupported",
			feature: "responseFormat",
			details: "Cursor replies in text",
		});

	let params = opts.params ?? [];
	if (
		opts.effort ||
		(options.reasoning ?? "provider-default") !== "provider-default"
	) {
		const status = await (await getHost()).status().catch(() => undefined);
		const model = status?.models?.find(
			(m) => m.id === modelId || m.aliases?.includes(modelId),
		);
		const effort = modelParams(model, options.reasoning, opts.effort, warnings);
		params = [
			...effort.filter((p) => !params.some((own) => own.id === p.id)),
			...params,
		];
	}
	const model = { id: modelId, ...(params.length && { params }) };

	const system = getSystem(options.prompt);
	const replace = opts.systemPromptPreset === null;
	const agent: CursorAgentOptions = {
		model,
		tools: builtins,
		disallowedTools: opts.disallowedTools,
		systemPrompt: replace ? system : undefined,
		mcpServers: opts.mcpServers,
		agents: opts.agents,
		mode: opts.mode,
		name: opts.name,
		cwd: opts.cwd,
		dirs: opts.additionalDirectories,
		settingSources: opts.settingSources,
		sandbox: opts.sandbox,
		autoReview: opts.autoReview,
	};
	for (const key of Object.keys(agent) as (keyof CursorAgentOptions)[])
		if (agent[key] === undefined) delete agent[key];

	const transcript = settings.transcript ?? defaultTranscript;
	const ctx: StepContext = {
		declared,
		tools,
		model,
		mode: opts.mode,
		agent,
		system: replace ? undefined : system,
		includeRawChunks: options.includeRawChunks ?? false,
		warnings,
		abortSignal: options.abortSignal,
		toMessage: (messages) =>
			toMessage(messages, {
				convertFile: settings.convertFile,
				warnings,
				transcript,
			}),
	};
	return { opts, ctx, transcript };
}

const orContinue = (message: CursorUserMessage): CursorUserMessage =>
	message.text || message.images?.length ? message : { text: "Continue." };

export function createCursorModel({
	modelId,
	defaults,
	settings,
	getHost,
	registry,
}: {
	modelId: CursorModelId;
	defaults?: CursorOptions;
	settings: CursorProviderSettings;
	getHost: () => Promise<CursorHost>;
	registry: SessionRegistry;
}): LanguageModelV4 {
	const hooks = {
		register: registry.register,
		unregister: registry.unregister,
		ttlMs: settings.sessionTtlMs ?? 10 * 60_000,
	};

	/** The agent a call goes on in: continued, resumed, or new. */
	const getSession = async (
		options: LanguageModelV4CallOptions,
		prepared: Awaited<ReturnType<typeof prepare>>,
	) => {
		const { ctx, opts, transcript } = prepared;
		const host = await getHost();
		const prompt = options.prompt;

		const cursor = findCursor(prompt);
		if (cursor) {
			const tail = getTail(prompt, cursor.index + 1);
			const live = registry.get(cursor.agentId);
			if (live?.canContinue(cursor.cursor)) {
				await live.update(ctx);
				await live.deliver(tail);
				return { session: live, route: "continue" as const };
			}
			// Saved where the conversation left it: the run it was left in
			// is the agent's last, and it finished.
			const last = await host.lastRun(cursor.agentId, { cwd: opts.cwd });
			if (last?.id === cursor.runId && last.status === "finished") {
				await live?.close();
				const message = await ctx.toMessage(tail.messages);
				const session = await Session.open(
					host,
					cursor.agentId,
					orContinue(message),
					ctx,
					hooks,
				);
				return { session, route: "resume" as const };
			}
		}

		// A new agent, told what came before.
		let start = prompt.length;
		while (start > 0 && prompt[start - 1].role === "user") start--;
		while (start > 0 && prompt[start - 1].role === "system") start--;
		const history = transcript(
			prompt.slice(0, start).filter((message) => message.role !== "system"),
		);
		const turn = prompt.slice(start).filter((m) => m.role === "user");
		const message = await ctx.toMessage(turn);
		const session = await Session.open(
			host,
			undefined,
			orContinue({
				...message,
				text: [history, message.text].filter(Boolean).join("\n\n"),
			}),
			ctx,
			hooks,
		);
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
				getHost,
			);
			const { session, route } = await getSession(options, prepared);
			prepared.ctx.route = route;
			return {
				stream: convertAsyncIteratorToReadableStream(
					session.stream(prepared.ctx),
				),
				request: { body: { route, options: prepared.ctx.agent } },
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
					case "custom":
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
