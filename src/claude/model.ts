import type {
	JSONObject,
	LanguageModelV4,
	LanguageModelV4CallOptions,
	LanguageModelV4Content,
	LanguageModelV4Prompt,
	LanguageModelV4StreamPart,
	SharedV4Warning,
} from "@ai-sdk/provider";
import {
	convertAsyncIteratorToReadableStream,
	createNullLanguageModelUsage,
	parseProviderOptions,
} from "@ai-sdk/provider-utils";
import type {
	EffortLevel,
	SDKUserMessage,
} from "@anthropic-ai/claude-agent-sdk";
import type {
	ClaudeCodeHost,
	ClaudeCodeRunOptions,
	ClaudeCodeToolDefinition,
} from "./host/protocol.ts";
import {
	type ClaudeCodeContentBlock,
	type ClaudeCodeModelId,
	ClaudeCodeOptions,
	type ClaudeCodeProviderSettings,
} from "./options.ts";
import {
	type Cursor,
	defaultTranscript,
	findCursor,
	getDangling,
	getSystem,
	getTail,
	METADATA_KEY,
	parseAnswers,
	type Tail,
	toBlocks,
	toToolContent,
} from "./prompt.ts";
import { Session, type StepContext } from "./session.ts";
import { ASK_USER_QUESTION, TOOL_ID_PREFIX } from "./tools.ts";

const PROVIDER = "claude-code";

type Reasoning = NonNullable<LanguageModelV4CallOptions["reasoning"]>;

const efforts: Record<Reasoning, EffortLevel | undefined> = {
	"provider-default": undefined,
	none: undefined,
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

/** The sessions a provider has live, by session id. */
export class SessionRegistry {
	private sessions = new Map<string, Session>();
	get = (id: string) => this.sessions.get(id);
	register = (session: Session) => {
		if (session.sessionId) this.sessions.set(session.sessionId, session);
	};
	unregister = (session: Session) => {
		if (session.sessionId && this.sessions.get(session.sessionId) === session)
			this.sessions.delete(session.sessionId);
	};
	all = () => [...this.sessions.values()];
}

/** A message the session starts with. */
const userMessage = (content: unknown[]): SDKUserMessage => ({
	type: "user",
	message: { role: "user", content: content as never },
	parent_tool_use_id: null,
});

const toolResultBlock = (
	id: string,
	content: ReturnType<typeof toToolContent>["content"] | string,
	isError: boolean,
) => ({
	type: "tool_result",
	tool_use_id: id,
	content:
		typeof content === "string"
			? content
			: content.map((part) =>
					part.type === "text"
						? part
						: {
								type: "image",
								source: {
									type: "base64",
									media_type: part.mimeType,
									data: part.data,
								},
							},
				),
	...(isError && { is_error: true }),
});

/**
 * The first message of a session forked at `cursor`: results for the calls
 * it left waiting there, then what the user said.
 */
function forkContent(
	prompt: LanguageModelV4Prompt,
	cursor: Cursor,
	tail: Tail,
	blocks: ClaudeCodeContentBlock[],
	questionKey: string | undefined,
) {
	const content: unknown[] = [];
	for (const call of getDangling(prompt, cursor)) {
		const result = tail.results.find((r) => r.toolCallId === call.id);
		const approval = tail.approvals.find((a) => a.approvalId === call.id);
		if (result && call.name === questionKey) {
			const answers = parseAnswers(result.output)?.answers;
			content.push(
				toolResultBlock(
					call.id,
					answers
						? `User has answered your questions: ${Object.entries(answers)
								.map(([q, a]) => `"${q}"="${a}"`)
								.join(
									", ",
								)}. You can now continue with the user's answers in mind.`
						: "The user did not answer.",
					!answers,
				),
			);
		} else if (result) {
			const { content: parts, isError } = toToolContent(result.output);
			content.push(toolResultBlock(call.id, parts, isError));
		} else if (approval?.approved)
			content.push(
				toolResultBlock(
					call.id,
					"The user approved this call, but the session restarted before it ran. Make it again if it is still needed.",
					true,
				),
			);
		else if (approval)
			content.push(
				toolResultBlock(
					call.id,
					`The user denied this tool call${approval.reason ? `: ${approval.reason}` : "."}`,
					true,
				),
			);
		else
			content.push(
				toolResultBlock(
					call.id,
					"This call did not run: the session restarted.",
					true,
				),
			);
	}
	content.push(...blocks);
	return content;
}

/** What a call to the model sets up, from its options. */
async function prepare(
	modelId: string,
	options: LanguageModelV4CallOptions,
	settings: ClaudeCodeProviderSettings,
	modelDefaults: ClaudeCodeOptions | undefined,
) {
	const warnings: SharedV4Warning[] = [];
	const own = await parseProviderOptions({
		provider: METADATA_KEY,
		providerOptions: options.providerOptions,
		schema: ClaudeCodeOptions,
	});
	const opts: ClaudeCodeOptions = {
		...settings.defaults,
		...modelDefaults,
		...own,
	};

	for (const setting of unsupportedSettings)
		if (options[setting] !== undefined)
			warnings.push({ type: "unsupported", feature: setting });

	// The call's tools: its own, served over MCP, and Claude Code's it declared.
	const declared = new Map<string, string>();
	let questionKey: string | undefined;
	let tools: ClaudeCodeToolDefinition[] = [];
	for (const tool of options.tools ?? []) {
		if (tool.type === "function")
			tools.push({
				name: tool.name,
				description: tool.description,
				inputSchema: tool.inputSchema as JSONObject,
			});
		else if (tool.id.startsWith(TOOL_ID_PREFIX)) {
			const name = tool.id.slice(TOOL_ID_PREFIX.length);
			if (name === ASK_USER_QUESTION) questionKey = tool.name;
			else declared.set(name, tool.name);
		} else
			warnings.push({
				type: "unsupported",
				feature: `provider tool ${tool.id}`,
			});
	}

	let builtins: ClaudeCodeRunOptions["tools"] =
		opts.builtinTools === "claude_code"
			? { type: "preset", preset: "claude_code" }
			: [
					...new Set([
						...(opts.builtinTools ?? []),
						...declared.keys(),
						...(questionKey ? [ASK_USER_QUESTION] : []),
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
			details: "Claude Code picks its own tools",
		});

	const disallowedTools = [
		...(opts.disallowedTools ?? []),
		// Without a way to answer, it can't ask.
		...(questionKey ? [] : [ASK_USER_QUESTION]),
	];

	let outputFormat: ClaudeCodeRunOptions["outputFormat"];
	if (options.responseFormat?.type === "json") {
		if (options.responseFormat.schema)
			outputFormat = {
				type: "json_schema",
				schema: options.responseFormat.schema as Record<string, unknown>,
			};
		else
			warnings.push({
				type: "unsupported",
				feature: "responseFormat without a schema",
			});
	}

	let flagSettings: ClaudeCodeRunOptions["settings"] = opts.settings as never;
	const flags = {
		...(opts.outputStyle !== undefined && { outputStyle: opts.outputStyle }),
		...(opts.fastMode !== undefined && { fastMode: opts.fastMode }),
	};
	if (Object.keys(flags).length) {
		if (typeof opts.settings === "string")
			warnings.push({
				type: "unsupported",
				feature: "outputStyle and fastMode with a settings file",
				details: "Put them in the file",
			});
		else flagSettings = { ...opts.settings, ...flags } as never;
	}

	const system = getSystem(options.prompt);
	const runOptions: ClaudeCodeRunOptions = {
		model: modelId,
		effort: opts.effort ?? efforts[options.reasoning ?? "provider-default"],
		thinking:
			opts.thinking ??
			(options.reasoning === "none" ? { type: "disabled" } : undefined),
		cwd: opts.cwd,
		additionalDirectories: opts.additionalDirectories,
		systemPrompt: opts.systemPromptPreset
			? {
					type: "preset",
					preset: opts.systemPromptPreset,
					append: system,
					excludeDynamicSections: opts.excludeDynamicSections,
				}
			: (system ?? ""),
		tools: builtins,
		allowedTools: opts.allowedTools,
		disallowedTools,
		settingSources: opts.settingSources ?? [],
		settings: flagSettings,
		mcpServers: opts.mcpServers,
		strictMcpConfig: opts.strictMcpConfig,
		permissionMode: opts.permissionMode,
		agent: opts.agent,
		agents: opts.agents,
		outputFormat,
		fallbackModel: opts.fallbackModel,
		maxTurns: opts.maxTurns,
		maxBudgetUsd: opts.maxBudgetUsd,
		persistSession: opts.persistSession,
		...opts.sdkOptions,
	};
	for (const key of Object.keys(runOptions) as (keyof ClaudeCodeRunOptions)[])
		if (runOptions[key] === undefined) delete runOptions[key];

	const transcript = settings.transcript ?? defaultTranscript;
	const ctx: StepContext = {
		declared,
		questionKey,
		mcpServerName: settings.mcpServerName ?? "ai-sdk",
		tools,
		model: modelId,
		permissionMode: opts.permissionMode,
		json: !!outputFormat,
		includeRawChunks: options.includeRawChunks ?? false,
		warnings,
		abortSignal: options.abortSignal,
		toBlocks: (messages) =>
			toBlocks(messages, {
				convertFile: settings.convertFile,
				warnings,
				transcript,
			}),
	};
	return { opts, runOptions, ctx, transcript };
}

export function createClaudeCodeModel({
	modelId,
	defaults,
	settings,
	getHost,
	registry,
}: {
	modelId: ClaudeCodeModelId;
	defaults?: ClaudeCodeOptions;
	settings: ClaudeCodeProviderSettings;
	getHost: () => Promise<ClaudeCodeHost>;
	registry: SessionRegistry;
}): LanguageModelV4 {
	const hooks = {
		register: registry.register,
		unregister: registry.unregister,
		ttlMs: settings.sessionTtlMs ?? 10 * 60_000,
	};

	/** The session a call goes on in: continued, forked, or new. */
	const getSession = async (
		options: LanguageModelV4CallOptions,
		prepared: Awaited<ReturnType<typeof prepare>>,
	) => {
		const { runOptions, ctx, opts, transcript } = prepared;
		const host = await getHost();
		const prompt = options.prompt;
		const request = {
			executable: opts.executable,
			tools: ctx.tools,
			mcpServerName: ctx.mcpServerName,
		};

		const cursor = findCursor(prompt);
		if (cursor) {
			const tail = getTail(prompt, cursor.index + 1);
			const live = registry.get(cursor.sessionId);
			if (live?.canContinue(cursor.cursor)) {
				await live.update(ctx);
				await live.deliver(tail);
				return { session: live, route: "continue" as const };
			}
			if (await host.hasSession(cursor.sessionId, { cwd: opts.cwd })) {
				const blocks = await ctx.toBlocks(tail.messages);
				const content = forkContent(
					prompt,
					cursor,
					tail,
					blocks,
					ctx.questionKey,
				);
				const session = await Session.start(
					host,
					{
						...request,
						options: {
							...runOptions,
							resume: cursor.sessionId,
							resumeSessionAt: cursor.cursor,
							forkSession: opts.forkSession ?? true,
						},
						prompt: userMessage(
							content.length ? content : [{ type: "text", text: "Continue." }],
						),
					},
					ctx,
					hooks,
				);
				return { session, route: "fork" as const };
			}
		}

		// A new session, told what came before.
		let start = prompt.length;
		while (start > 0 && prompt[start - 1].role === "user") start--;
		while (start > 0 && prompt[start - 1].role === "system") start--;
		const history = transcript(
			prompt.slice(0, start).filter((message) => message.role !== "system"),
		);
		const turn = prompt.slice(start).filter((m) => m.role === "user");
		const blocks = await ctx.toBlocks(turn);
		const content: unknown[] = [
			...(history ? [{ type: "text", text: history }] : []),
			...blocks,
		];
		if (!blocks.length) content.push({ type: "text", text: "Continue." });
		const session = await Session.start(
			host,
			{ ...request, options: runOptions, prompt: userMessage(content) },
			ctx,
			hooks,
		);
		return { session, route: "new" as const };
	};

	return {
		specificationVersion: "v4",
		provider: PROVIDER,
		modelId,
		supportedUrls: {
			"image/*": [/^https?:\/\//],
			"application/pdf": [/^https?:\/\//],
		},

		async doStream(options) {
			const prepared = await prepare(modelId, options, settings, defaults);
			const { session, route } = await getSession(options, prepared);
			prepared.ctx.route = route;
			return {
				stream: convertAsyncIteratorToReadableStream(
					session.stream(prepared.ctx),
				),
				request: { body: { route, options: prepared.runOptions } },
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
