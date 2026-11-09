import type {
	JSONObject,
	JSONValue,
	LanguageModelV4FinishReason,
	LanguageModelV4StreamPart,
	LanguageModelV4Usage,
	SharedV4ProviderMetadata,
	SharedV4Warning,
} from "@ai-sdk/provider";
import type { InteractionUpdate } from "@cursor/sdk";
import type {
	CursorAgentOptions,
	CursorHost,
	CursorHostEvent,
	CursorModelSelection,
	CursorTokenUsage,
	CursorToolAnswer,
	CursorToolDefinition,
	CursorUserMessage,
} from "./host/protocol.ts";
import { METADATA_KEY, type Tail, toToolAnswer, withSystem } from "./prompt.ts";

/** What an agent is opened with, but what each run can change. */
const keyOf = (options: CursorAgentOptions) =>
	JSON.stringify({ ...options, model: undefined, mode: undefined });

/** The MCP server the SDK serves the caller's tools under. */
const CUSTOM_TOOLS_SERVER = "custom-user-tools";

/** A step's end, as {@link LanguageModelV4StreamPart} metadata says what it waits on. */
export type CursorAwaiting =
	/** The caller's tools to answer. */
	| "tools"
	/** Nothing: the run goes on, to be streamed by the next call. */
	| "continue";

/** A stream's `error`, from Cursor. */
export class CursorError extends Error {
	override name = "CursorError";
}

/** What a step needs of the call it streams for. */
export interface StepContext {
	/** Cursor's names of the built-ins the call declared, to their keys. */
	declared: Map<string, string>;
	tools: CursorToolDefinition[];
	model: CursorModelSelection;
	mode?: "agent" | "plan";
	/** What the agent is opened with; a change reopens it between runs. */
	agent: CursorAgentOptions;
	/** System instructions to put at the top of a message, when they change. */
	system?: string;
	includeRawChunks: boolean;
	warnings: SharedV4Warning[];
	abortSignal?: AbortSignal;
	toMessage: (messages: Tail["messages"]) => Promise<CursorUserMessage>;
	/** How the step's call found its agent: continued, resumed, or new. */
	route?: "continue" | "resume" | "new";
}

interface Call {
	id: string;
	/** What Cursor calls the tool. */
	name: string;
	/** What the call's caller calls it. */
	toolName: string;
	kind: "client" | "builtin";
	/** Declared by no call: a dynamic tool. */
	dynamic: boolean;
	input: unknown;
	/**
	 * `open` while Cursor runs it; `waiting` on the caller for a result;
	 * `done` once answered or its result was streamed.
	 */
	state: "open" | "waiting" | "done";
}

/** A subagent's work so far, as a `task` call's preliminary results. */
export interface CursorSubagentProgress {
	type: "subagent";
	messages: CursorSubagentMessage[];
}

export type CursorSubagentMessage =
	| { role: "assistant"; text: string }
	| {
			role: "tool";
			toolName: string;
			input: unknown;
			output?: unknown;
			isError?: boolean;
	  };

/** A step's own state. */
interface Step {
	/** The model call's tools were all asked for. */
	listed: boolean;
	/** Tool calls the step streamed. */
	calls: number;
	result?: Extract<CursorHostEvent, { type: "result" }>;
	usage?: CursorTokenUsage;
	/** The text or reasoning block streaming. */
	block?: { id: string; kind: "text" | "reasoning" };
	aborted: boolean;
	/** The step's model call is done, with all of its built-ins. */
	done: boolean;
}

const usageOf = (usage: CursorTokenUsage | undefined): LanguageModelV4Usage => {
	if (!usage)
		return {
			inputTokens: {
				total: undefined,
				noCache: undefined,
				cacheRead: undefined,
				cacheWrite: undefined,
			},
			outputTokens: { total: undefined, text: undefined, reasoning: undefined },
		};
	return {
		inputTokens: {
			total: usage.inputTokens + usage.cacheReadTokens + usage.cacheWriteTokens,
			noCache: usage.inputTokens,
			cacheRead: usage.cacheReadTokens,
			cacheWrite: usage.cacheWriteTokens,
		},
		outputTokens: {
			total: usage.outputTokens,
			text:
				usage.reasoningTokens === undefined
					? undefined
					: usage.outputTokens - usage.reasoningTokens,
			reasoning: usage.reasoningTokens,
		},
		raw: usage as unknown as JSONObject,
	};
};

const json = (value: unknown): JSONValue =>
	value === undefined ? null : JSON.parse(JSON.stringify(value));

type ToolCallUpdate = Extract<
	InteractionUpdate,
	{ type: "tool-call-started" | "tool-call-completed" }
>;

/** What a call of Cursor's is, from the tool call the SDK reports. */
const describe = (toolCall: ToolCallUpdate["toolCall"]) => {
	const call = toolCall as { type: string; args?: any; result?: any };
	if (
		call.type === "mcp" &&
		call.args?.providerIdentifier === CUSTOM_TOOLS_SERVER
	)
		return {
			kind: "client" as const,
			name: String(call.args.toolName),
			input: call.args.args ?? {},
		};
	return { kind: "builtin" as const, name: call.type, input: call.args ?? {} };
};

/**
 * One Cursor agent, open in a host: its runs streamed a step at a time, and
 * waiting between steps for what the next call brings. What happens while no
 * step streams is kept for the next.
 */
export class Session {
	readonly agentId: string;
	/** Where the conversation is: `<runId>#<step>`. */
	cursor?: string;
	ended = false;
	attached = false;

	private runId?: string;
	private running = false;
	private events: CursorHostEvent[] = [];
	private wake?: () => void;
	private calls = new Map<string, Call>();
	/** The caller's tool calls the host asked about. */
	private questions = new Set<string>();
	/** Answers given before the host asked. */
	private early = new Map<string, CursorToolAnswer>();
	/** What the user said while a run couldn't take it, for the next run. */
	private queued: CursorUserMessage[] = [];
	/** `task` calls' progress, by call. */
	private subagents = new Map<string, CursorSubagentMessage[]>();
	private stepCursors = new Set<string>();
	private step = 0;
	private blocks = 0;
	private ctx: StepContext;
	private agentKey: string;
	/** The system instructions last sent to the agent. */
	private system?: string;
	private error?: string;
	private timer?: ReturnType<typeof setTimeout>;

	private readonly host: CursorHost;
	private readonly hooks: {
		register: (session: Session) => void;
		unregister: (session: Session) => void;
		ttlMs: number;
	};

	private constructor(
		host: CursorHost,
		agentId: string,
		ctx: StepContext,
		hooks: Session["hooks"],
		system: string | undefined,
	) {
		this.host = host;
		this.agentId = agentId;
		this.ctx = ctx;
		this.agentKey = keyOf(ctx.agent);
		this.hooks = hooks;
		this.system = system;
		hooks.register(this);
	}

	/**
	 * Opens an agent in `host`, new or saved, and sends it its first message.
	 * `system` is what a saved agent was last told, if it was.
	 */
	static async open(
		host: CursorHost,
		agentId: string | undefined,
		message: CursorUserMessage,
		ctx: StepContext,
		hooks: Session["hooks"],
		system?: string,
	) {
		const id = await host.open({ agentId, options: ctx.agent });
		const session = new Session(host, id, ctx, hooks, system);
		await session.start(message);
		return session;
	}

	/** Whether a call that left the conversation at `cursor` continues here. */
	canContinue(cursor: string) {
		return !this.ended && !this.attached && this.stepCursors.has(cursor);
	}

	/** Brings what a call changed into the agent, between its runs. */
	async update(ctx: StepContext) {
		this.ctx = ctx;
		const key = keyOf(ctx.agent);
		if (key === this.agentKey) return;
		if (this.running) {
			ctx.warnings.push({
				type: "other",
				message:
					"Cursor takes changed instructions, tools or settings from its next run on.",
			});
			return;
		}
		await this.host.open({ agentId: this.agentId, options: ctx.agent });
		this.agentKey = key;
	}

	/** The message to send, with system instructions it hasn't had. */
	private withSystem(message: CursorUserMessage): CursorUserMessage {
		const system = this.ctx.system;
		if (!system || system === this.system) return message;
		this.system = system;
		return { ...message, text: withSystem(system, message.text) };
	}

	/** Starts a run with `message`. */
	private async start(message: CursorUserMessage) {
		this.running = true;
		try {
			this.runId = await this.host.send(
				this.agentId,
				{
					message: this.withSystem(message),
					model: this.ctx.model,
					mode: this.ctx.mode,
					tools: this.ctx.tools,
				},
				(event) => this.push(event),
			);
		} catch (error) {
			this.running = false;
			throw error;
		}
	}

	/**
	 * Gives the agent what came after its cursor: results for the caller's
	 * calls it waits on, then what the user said. Calls it waits on that the
	 * call left unanswered are refused.
	 */
	async deliver(tail: Tail) {
		for (const result of tail.results) {
			const call = this.calls.get(result.toolCallId);
			if (call?.kind !== "client" || call.state === "done") continue;
			await this.answer(call, toToolAnswer(result.output));
		}
		for (const call of this.calls.values())
			if (call.state === "waiting")
				await this.answer(call, {
					content: [{ type: "text", text: "No result was given." }],
					isError: true,
				});
		const message = await this.ctx.toMessage(tail.messages);
		if (!message.text && !message.images?.length) return;
		if (!this.running) return this.start(message);
		// Text can steer the run; images wait for the next.
		if (
			!message.images?.length &&
			(!this.ctx.system || this.ctx.system === this.system) &&
			(await this.host.steer(this.agentId, message.text))
		)
			return;
		this.queued.push(message);
	}

	/** Answers one of the caller's calls, or keeps it until the host asks. */
	private async answer(call: Call, answer: CursorToolAnswer) {
		call.state = "done";
		if (this.questions.delete(call.id))
			await this.host.answer(this.agentId, call.id, answer);
		else this.early.set(call.id, answer);
	}

	/** Takes an event from the host, answering at once what was answered early. */
	private push(event: CursorHostEvent) {
		if (event.type === "tool-call") {
			const early = this.early.get(event.toolCallId);
			if (early) {
				this.early.delete(event.toolCallId);
				void this.host
					.answer(this.agentId, event.toolCallId, early)
					.catch(() => {});
			} else this.questions.add(event.toolCallId);
			return;
		}
		this.events.push(event);
		this.wake?.();
	}

	private next(signal?: AbortSignal) {
		return new Promise<CursorHostEvent | "abort">((resolve) => {
			if (signal?.aborted) return resolve("abort");
			const event = this.events.shift();
			if (event) return resolve(event);
			const done = (value: CursorHostEvent | "abort") => {
				signal?.removeEventListener("abort", onAbort);
				this.wake = undefined;
				resolve(value);
			};
			const onAbort = () => done("abort");
			signal?.addEventListener("abort", onAbort, { once: true });
			this.wake = () => {
				const event = this.events.shift();
				if (event) done(event);
			};
		});
	}

	/** Closes the agent; it can still be resumed from where it was saved. */
	async close() {
		clearTimeout(this.timer);
		if (this.ended) return;
		this.ended = true;
		this.hooks.unregister(this);
		await this.host.close(this.agentId).catch(() => {});
	}

	private metadata(extra?: Record<string, unknown>): SharedV4ProviderMetadata {
		if (this.cursor) this.stepCursors.add(this.cursor);
		return {
			[METADATA_KEY]: {
				agentId: this.agentId,
				...(this.cursor && { cursor: this.cursor }),
				...((extra ?? {}) as JSONObject),
			},
		};
	}

	/** Built-ins of the step still running. */
	private openCalls() {
		let open = 0;
		for (const call of this.calls.values())
			if (call.kind === "builtin" && call.state === "open") open++;
		return open;
	}

	/** Streams one step: what is buffered, then on until the step ends. */
	async *stream(ctx: StepContext): AsyncGenerator<LanguageModelV4StreamPart> {
		this.ctx = ctx;
		this.attached = true;
		this.stepCursors.clear();
		clearTimeout(this.timer);
		this.step++;
		this.cursor = this.runId ? `${this.runId}#${this.step}` : undefined;
		const step: Step = {
			listed: false,
			calls: 0,
			aborted: false,
			done: false,
		};
		yield { type: "stream-start", warnings: ctx.warnings };
		yield {
			type: "response-metadata",
			id: this.runId,
			modelId: ctx.model.id,
			timestamp: new Date(),
		};
		try {
			while (!this.ended && !step.done) {
				if (!this.running && !this.events.length) break;
				const event = await this.next(ctx.abortSignal);
				if (event === "abort") {
					step.aborted = true;
					break;
				}
				yield* this.handle(event, step);
				// Cursor can wait for a custom result before starting another
				// call or announcing the batch. Hand it back immediately; the
				// next step collects whatever comes after it, including built-ins
				// still open. The batch's count can include invisible SDK calls.
				if (
					[...this.calls.values()].some((call) => call.state === "waiting") ||
					(step.calls && step.listed && !this.openCalls())
				)
					step.done = true;
			}
			yield* this.finish(step);
		} finally {
			this.attached = false;
			if (this.cursor) this.stepCursors.add(this.cursor);
			if (!this.ended) {
				this.timer = setTimeout(() => void this.close(), this.hooks.ttlMs);
				(this.timer as { unref?: () => void }).unref?.();
			}
		}
	}

	private *endBlock(step: Step): Generator<LanguageModelV4StreamPart> {
		if (!step.block) return;
		const { id, kind } = step.block;
		step.block = undefined;
		yield { type: `${kind}-end`, id, providerMetadata: this.metadata() };
	}

	private *delta(
		step: Step,
		kind: "text" | "reasoning",
		text: string,
	): Generator<LanguageModelV4StreamPart> {
		if (!text) return;
		if (step.block?.kind !== kind) {
			yield* this.endBlock(step);
			step.block = { id: `${this.runId}:${this.blocks++}`, kind };
			yield { type: `${kind}-start`, id: step.block.id };
		}
		yield { type: `${kind}-delta`, id: step.block.id, delta: text };
	}

	private async *handle(
		event: CursorHostEvent,
		step: Step,
	): AsyncGenerator<LanguageModelV4StreamPart> {
		switch (event.type) {
			case "delta":
				if (this.ctx.includeRawChunks)
					yield { type: "raw", rawValue: event.update };
				yield* this.read(event.update, step);
				return;
			case "result":
				step.result = event;
				step.usage ??= event.usage;
				return;
			case "error":
				this.error = event.message;
				return;
			case "end": {
				this.running = false;
				// What the user said meanwhile starts the next run, in this step.
				const queued = this.queued.splice(0);
				if (queued.length && !step.aborted && !this.ended) {
					step.result = undefined;
					step.usage = undefined;
					yield* this.endBlock(step);
					await this.start({
						text: queued.map((message) => message.text).join("\n\n"),
						images: queued.flatMap((message) => message.images ?? []),
					});
					this.cursor = `${this.runId}#${this.step}`;
				} else step.done = true;
				return;
			}
		}
	}

	private register(update: ToolCallUpdate) {
		const existing = this.calls.get(update.callId);
		if (existing) return existing;
		const { kind, name, input } = describe(update.toolCall);
		const call: Call = {
			id: update.callId,
			name,
			toolName:
				kind === "client" ? name : (this.ctx.declared.get(name) ?? name),
			kind,
			dynamic: kind === "builtin" && !this.ctx.declared.has(name),
			input,
			state: kind === "client" ? "waiting" : "open",
		};
		this.calls.set(call.id, call);
		return call;
	}

	private *read(
		update: InteractionUpdate,
		step: Step,
	): Generator<LanguageModelV4StreamPart> {
		// Not typed by the SDK: the model call's tool calls were all made.
		if ((update as { type: string }).type === "tool-requests-listed") {
			// Remember announcements before calls start as well. Calls only
			// Cursor sees end no step: the stream also requires a visible call.
			step.listed = true;
			yield* this.endBlock(step);
			return;
		}
		switch (update.type) {
			case "text-delta":
				yield* this.delta(step, "text", update.text);
				return;
			case "thinking-delta":
				yield* this.delta(step, "reasoning", update.text);
				return;
			case "thinking-completed":
				if (step.block?.kind === "reasoning") yield* this.endBlock(step);
				return;
			case "tool-call-started": {
				yield* this.endBlock(step);
				const call = this.register(update);
				step.calls++;
				yield {
					type: "tool-call",
					toolCallId: call.id,
					toolName: call.toolName,
					input: JSON.stringify(call.input ?? {}),
					...(call.kind === "builtin" && { providerExecuted: true }),
					...(call.dynamic && { dynamic: true }),
					providerMetadata: this.metadata(),
				};
				return;
			}
			case "tool-call-completed": {
				const call = this.register(update);
				if (call.kind === "client" || call.state === "done") {
					call.state = "done";
					return;
				}
				call.state = "done";
				const result = (update.toolCall as { result?: { status?: string } })
					.result;
				yield {
					type: "tool-result",
					toolCallId: call.id,
					toolName: call.toolName,
					result: json(result) as NonNullable<JSONValue>,
					...(result?.status &&
						result.status !== "success" && { isError: true }),
					...(call.dynamic && { dynamic: true }),
					providerMetadata: this.metadata(),
				};
				return;
			}
			case "tool-call-delta":
				yield* this.readSubagent(update.callId, update.taskUpdate);
				return;
			case "turn-ended":
				yield* this.endBlock(step);
				if (update.usage)
					step.usage = {
						...update.usage,
						reasoningTokens: update.usage.reasoningTokens,
					};
				return;
			case "summary-completed":
				yield {
					type: "custom",
					kind: "cursor.compaction",
					providerMetadata: this.metadata(),
				};
				return;
		}
	}

	private *readSubagent(
		parent: string,
		update: Extract<
			InteractionUpdate,
			{ type: "tool-call-delta" }
		>["taskUpdate"],
	): Generator<LanguageModelV4StreamPart> {
		const messages = this.subagents.get(parent) ?? [];
		this.subagents.set(parent, messages);
		const last = messages.at(-1);
		if (update.type === "text-delta") {
			if (last?.role === "assistant") last.text += update.text;
			else messages.push({ role: "assistant", text: update.text });
			return;
		}
		if (update.type === "tool-call-started") {
			const { name, input } = describe(update.toolCall);
			messages.push({ role: "tool", toolName: name, input });
		} else if (update.type === "tool-call-completed") {
			const { name, input } = describe(update.toolCall);
			const result = (update.toolCall as { result?: { status?: string } })
				.result;
			const entry = messages.findLast(
				(message) =>
					message.role === "tool" &&
					message.toolName === name &&
					message.output === undefined,
			);
			const done = {
				role: "tool" as const,
				toolName: name,
				input,
				output: result,
				...(result?.status && result.status !== "success" && { isError: true }),
			};
			if (entry) Object.assign(entry, done);
			else messages.push(done);
		} else if (update.type !== "step-completed") return;
		const call = this.calls.get(parent);
		yield {
			type: "tool-result",
			toolCallId: parent,
			toolName: call?.toolName ?? "task",
			result: json({
				type: "subagent",
				messages,
			} satisfies CursorSubagentProgress) as NonNullable<JSONValue>,
			preliminary: true,
			...(call?.dynamic && { dynamic: true }),
		};
	}

	private async *finish(step: Step): AsyncGenerator<LanguageModelV4StreamPart> {
		yield* this.endBlock(step);

		if (step.aborted) {
			for (const call of this.calls.values())
				if (call.state !== "done") call.state = "done";
			this.questions.clear();
			this.early.clear();
			this.queued = [];
			if (this.running) await this.host.cancel(this.agentId).catch(() => {});
		}

		const result = step.result;
		const awaiting: CursorAwaiting[] = [];
		let unified: LanguageModelV4FinishReason["unified"];
		let error = this.error;
		if (step.aborted) unified = "other";
		else if (result) {
			unified =
				result.status === "finished"
					? "stop"
					: result.status === "cancelled"
						? "other"
						: "error";
			if (result.status === "error")
				error ??= result.error ?? "The run failed.";
		} else if (
			step.listed ||
			[...this.calls.values()].some((call) => call.state === "waiting")
		) {
			unified = "tool-calls";
			awaiting.push(
				[...this.calls.values()].some((call) => call.state === "waiting")
					? "tools"
					: "continue",
			);
		} else if (this.ended || (!this.running && step.done)) {
			unified = "error";
			error ??= "Cursor ended unexpectedly.";
		} else unified = "stop";
		if (error) {
			this.error = undefined;
			yield { type: "error", error: new CursorError(error) };
		}

		yield {
			type: "custom",
			kind: "cursor.step",
			providerMetadata: this.metadata({ awaiting }),
		};
		yield {
			type: "finish",
			finishReason: { unified, raw: result?.status },
			usage: usageOf(step.usage),
			providerMetadata: this.metadata({
				awaiting,
				...(this.runId && { runId: this.runId }),
				...(this.ctx.route && { route: this.ctx.route }),
				...(result?.requestId && { requestId: result.requestId }),
				...(result?.durationMs !== undefined && {
					durationMs: result.durationMs,
				}),
				...(result?.model && { model: json(result.model) }),
			}),
		};
	}
}
