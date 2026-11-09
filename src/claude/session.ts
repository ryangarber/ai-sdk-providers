import type {
	JSONObject,
	JSONValue,
	LanguageModelV4FinishReason,
	LanguageModelV4StreamPart,
	LanguageModelV4Usage,
	SharedV4ProviderMetadata,
	SharedV4Warning,
} from "@ai-sdk/provider";
import type {
	SDKMessage,
	SDKResultMessage,
	SDKUserMessage,
} from "@anthropic-ai/claude-agent-sdk";
import type {
	ClaudeCodeHost,
	ClaudeCodeHostAnswer,
	ClaudeCodeHostEvent,
	ClaudeCodeRunRequest,
	ClaudeCodeRunUpdate,
	ClaudeCodeToolDefinition,
} from "./host/protocol.ts";
import type { ClaudeCodeContentBlock } from "./options.ts";
import {
	METADATA_KEY,
	parseAnswers,
	type Tail,
	toToolContent,
} from "./prompt.ts";
import {
	ASK_USER_QUESTION,
	type ClaudeCodeResultBlock,
	type ClaudeCodeSubagentMessage,
	type ClaudeCodeToolOutput,
} from "./tools.ts";

/** How long a step waits, once only the caller can move it on, for anything else. */
const SETTLE_MS = 50;
/**
 * How long after a result to look for the query a background task's
 * notification starts, when no query has taken it in yet. Claude Code starts
 * it at once; one that never comes was taken in by the query that ended.
 */
const OWED_MS = 1000;

/** A step's end, as {@link LanguageModelV4StreamPart} metadata says what it waits on. */
export type ClaudeCodeAwaiting =
	/** The caller's tools, and questions, to answer. */
	| "tools"
	/** Approvals of Claude Code's own calls. */
	| "approval"
	/** Nothing: the session goes on, to be streamed by the next call. */
	| "continue"
	/** Work in the background, which goes on after the turn ended. */
	| "background";

/** A stream's `error`, from Claude Code. */
export class ClaudeCodeError extends Error {
	override name = "ClaudeCodeError";
	/** The result's subtype, e.g. `error_max_turns`, if there was one. */
	readonly subtype?: string;
	constructor(message: string, subtype?: string) {
		super(message);
		this.subtype = subtype;
	}
}

/** What a step needs of the call it streams for. */
export interface StepContext {
	/** Claude Code's names of the built-ins the call declared, to their keys. */
	declared: Map<string, string>;
	/** The call's key for AskUserQuestion, if it declared it. */
	questionKey?: string;
	mcpServerName: string;
	tools: ClaudeCodeToolDefinition[];
	model: string;
	permissionMode?: string;
	/** Whether `structured_output` is the reply, as text. */
	json: boolean;
	includeRawChunks: boolean;
	warnings: SharedV4Warning[];
	abortSignal?: AbortSignal;
	toBlocks: (messages: Tail["messages"]) => Promise<ClaudeCodeContentBlock[]>;
	/** How the step's call found its session: continued, forked, or new. */
	route?: "continue" | "fork" | "new";
}

interface Call {
	id: string;
	/** What Claude Code calls the tool. */
	name: string;
	/** What the call's caller calls it. */
	toolName: string;
	kind: "client" | "question" | "builtin";
	/** Declared by no call: a dynamic tool. */
	dynamic: boolean;
	input: unknown;
	/** The subagent's call it was made under, if any. */
	parent?: string;
	/**
	 * `open` while Claude Code has it; `waiting` on the caller, for a result or
	 * an approval; `done` once answered or its result was streamed.
	 */
	state: "open" | "waiting" | "done";
	/** The step its call was last streamed in. */
	announced?: number;
}

/** A step's own state. */
interface Step {
	sawTurn: boolean;
	turnOpen: boolean;
	stopReason?: string;
	usage: { start?: Record<string, any>; delta?: Record<string, any> };
	result?: SDKResultMessage;
	/** Text and reasoning blocks streaming, by id. */
	blocks: Map<string, "text" | "reasoning">;
	/** Tool inputs streaming. */
	inputs: Set<string>;
	aborted: boolean;
	/** Claude Code goes on after its result, with a query of its own. */
	more?: boolean;
}

const usageOf = ({ start, delta }: Step["usage"]): LanguageModelV4Usage => {
	const pick = (key: string): number | undefined =>
		delta?.[key] ?? start?.[key];
	const noCache = pick("input_tokens");
	const cacheRead = pick("cache_read_input_tokens");
	const cacheWrite = pick("cache_creation_input_tokens");
	const known = [noCache, cacheRead, cacheWrite].filter(
		(value) => value !== undefined,
	) as number[];
	return {
		inputTokens: {
			total: known.length ? known.reduce((a, b) => a + b, 0) : undefined,
			noCache,
			cacheRead,
			cacheWrite,
		},
		outputTokens: {
			total: delta?.output_tokens,
			text: undefined,
			reasoning: undefined,
		},
		raw: start || delta ? ({ ...start, ...delta } as JSONObject) : undefined,
	};
};

const finishReasons: Record<string, LanguageModelV4FinishReason["unified"]> = {
	end_turn: "stop",
	stop_sequence: "stop",
	max_tokens: "length",
	model_context_window_exceeded: "length",
	refusal: "content-filter",
	tool_use: "tool-calls",
	pause_turn: "tool-calls",
};

/** A tool result's content, as Claude Code reports it, as a result. */
const toOutput = (content: unknown): ClaudeCodeToolOutput => {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content.flatMap((block: any): ClaudeCodeResultBlock[] => {
		if (block?.type === "text" && typeof block.text === "string")
			return [{ type: "text", text: block.text }];
		if (block?.type === "image" && block.source?.type === "base64")
			return [
				{
					type: "image",
					mediaType: block.source.media_type ?? "image/png",
					data: block.source.data ?? "",
				},
			];
		if (block?.type === "document" && block.source?.type === "base64")
			return [
				{
					type: "document",
					mediaType: block.source.media_type ?? "application/pdf",
					data: block.source.data ?? "",
					...(typeof block.title === "string" && { title: block.title }),
				},
			];
		return [];
	});
};

const json = (value: unknown): JSONValue =>
	value === undefined ? null : JSON.parse(JSON.stringify(value));

/** What the caller answered an AskUserQuestion with, as its updated input. */
const toAnswers = (
	input: unknown,
	output: Tail["results"][number]["output"],
): ClaudeCodeHostAnswer => {
	const parsed = parseAnswers(output);
	if (!parsed)
		return {
			type: "permission",
			result: {
				behavior: "deny",
				message:
					output.type === "execution-denied"
						? (output.reason ?? "The user did not answer.")
						: "The user did not answer.",
			},
		};
	return {
		type: "permission",
		result: {
			behavior: "allow",
			updatedInput: { ...(input as Record<string, unknown>), ...parsed },
		},
	};
};

/**
 * One Claude Code session, live in a host: streamed a step at a time, and
 * waiting between steps for what the next call brings. What happens while no
 * step streams is kept for the next.
 */
export class Session {
	sessionId?: string;
	/** The session's last saved entry seen. */
	cursor?: string;
	/** Where the last step left it, which a call must resume at to continue. */
	stepCursor?: string;
	/** Every place the last step said it left the session at. */
	private stepCursors = new Set<string>();
	ended = false;
	attached = false;
	/** Done with its turns until it is sent more, with nothing in the background. */
	private idle = false;

	private run?: string;
	private error?: string;
	private events: ClaudeCodeHostEvent[] = [];
	private wake?: () => void;
	private calls = new Map<string, Call>();
	/** Host questions waiting on the caller, by call: tool calls and asks. */
	private questions = new Map<string, string>();
	/** Answers given before the host asked. */
	private early = new Map<string, ClaudeCodeHostAnswer>();
	/** The calls asked about, and what was asked. */
	private asks = new Map<
		string,
		Extract<ClaudeCodeHostEvent, { type: "permission" }>
	>();
	/** Built-ins running, by call. */
	private running = new Set<string>();
	/** A background task's notification no query has taken in yet. */
	private owed = false;
	/** Calls running in the background, their results placeholders. */
	private background = new Set<string>();
	/**
	 * A turn was interrupted: its query's result, an error saying it stopped, is
	 * the aborted step's and not the next one's. `sent` once the caller sent
	 * more, which a query of its own takes in after it.
	 */
	private interrupted?: { sent: boolean };
	/**
	 * A notification of a task from a run before this one, which a resumed
	 * session replays: the query it starts ends at once, with no turns.
	 */
	private stale = false;
	/** Reports subagents handed back, by the call that started them. */
	private handbacks = new Map<string, string>();
	/** The calls tasks run for, by task. */
	private tasks = new Map<string, string>();
	private subagents = new Map<string, ClaudeCodeSubagentMessage[]>();
	/** Messages streamed in parts, by id, whose whole messages add nothing. */
	private streamed = new Set<string>();
	private step = 0;
	private ctx!: StepContext;
	private config?: { tools: string; model: string; permissionMode?: string };
	private timer?: ReturnType<typeof setTimeout>;

	private readonly host: ClaudeCodeHost;
	private readonly hooks: {
		register: (session: Session) => void;
		unregister: (session: Session) => void;
		ttlMs: number;
	};

	private constructor(host: ClaudeCodeHost, hooks: Session["hooks"]) {
		this.host = host;
		this.hooks = hooks;
	}

	/** Starts a session in `host`. */
	static async start(
		host: ClaudeCodeHost,
		request: ClaudeCodeRunRequest,
		ctx: StepContext,
		hooks: Session["hooks"],
	) {
		const session = new Session(host, hooks);
		session.ctx = ctx;
		session.config = Session.configOf(ctx);
		session.run = await host.start(request, (event) => session.push(event));
		return session;
	}

	private static configOf(ctx: StepContext) {
		return {
			tools: JSON.stringify(ctx.tools),
			model: ctx.model,
			permissionMode: ctx.permissionMode,
		};
	}

	/** Whether a call that left the conversation at `cursor` continues here. */
	canContinue(cursor: string) {
		return !this.ended && !this.attached && this.stepCursors.has(cursor);
	}

	/** Brings what a call changed into the live session. */
	async update(ctx: StepContext) {
		this.ctx = ctx;
		const next = Session.configOf(ctx);
		const update: ClaudeCodeRunUpdate = {};
		if (next.tools !== this.config?.tools) update.tools = ctx.tools;
		if (next.model !== this.config?.model) update.model = ctx.model;
		if (
			next.permissionMode &&
			next.permissionMode !== this.config?.permissionMode
		)
			update.permissionMode =
				ctx.permissionMode as ClaudeCodeRunUpdate["permissionMode"];
		this.config = next;
		if (Object.keys(update).length && this.run)
			await this.host.update(this.run, update);
	}

	/**
	 * Gives the session what came after its cursor: results, answers and
	 * approvals for what it waits on, then what the user said. Anything it
	 * waits on that the call left unanswered is refused.
	 */
	async deliver(tail: Tail) {
		for (const result of tail.results) {
			const call = this.calls.get(result.toolCallId);
			if (!call || call.state === "done") continue;
			if (call.kind === "question")
				await this.answer(call, toAnswers(call.input, result.output));
			else if (call.kind === "client") {
				const { content, isError } = toToolContent(result.output);
				await this.answer(call, { type: "tool-result", content, isError });
			}
		}
		for (const approval of tail.approvals) {
			const call = this.calls.get(approval.approvalId);
			if (!call || !this.asks.has(call.id)) continue;
			const ask = this.asks.get(call.id);
			this.asks.delete(call.id);
			await this.answer(
				call,
				{
					type: "permission",
					result: approval.approved
						? { behavior: "allow", updatedInput: ask?.input }
						: {
								behavior: "deny",
								message: approval.reason ?? "The user denied this tool call.",
							},
				},
				"open",
			);
		}
		for (const call of this.calls.values()) {
			if (call.state !== "waiting") continue;
			this.asks.delete(call.id);
			await this.answer(
				call,
				call.kind === "client"
					? {
							type: "tool-result",
							content: [{ type: "text", text: "No result was given." }],
							isError: true,
						}
					: {
							type: "permission",
							result: { behavior: "deny", message: "The user did not answer." },
						},
				call.kind === "builtin" ? "open" : "done",
			);
		}
		const content = await this.ctx.toBlocks(tail.messages);
		// Picked up with nothing new after it stopped, as after an abort: it goes
		// on as it would have, as a new session told nothing new does.
		if (
			!content.length &&
			(this.interrupted || (this.idle && !this.background.size && !this.owed))
		)
			content.push({ type: "text", text: "Continue." });
		if (content.length) {
			this.idle = false;
			if (this.interrupted) this.interrupted.sent = true;
		}
		if (content.length && this.run)
			await this.host.send(this.run, {
				type: "user",
				message: { role: "user", content: content as never },
				parent_tool_use_id: null,
				priority: "next",
			} satisfies SDKUserMessage);
	}

	/** Answers what the host asked about a call, or keeps it until it asks. */
	private async answer(
		call: Call,
		answer: ClaudeCodeHostAnswer,
		state: Call["state"] = "done",
	) {
		call.state = state;
		const question = this.questions.get(call.id);
		this.questions.delete(call.id);
		if (question && this.run)
			await this.host.answer(this.run, question, answer);
		else this.early.set(call.id, answer);
	}

	/** Takes an event from the host, answering at once what was answered early. */
	private push(event: ClaudeCodeHostEvent) {
		if (event.type === "tool-call" || event.type === "permission") {
			const id = event.toolUseId;
			const early = id ? this.early.get(id) : undefined;
			if (id && early && this.run) {
				this.early.delete(id);
				void this.host.answer(this.run, event.id, early);
				return;
			}
		}
		this.events.push(event);
		this.wake?.();
	}

	private next(ms: number | undefined, signal?: AbortSignal) {
		return new Promise<ClaudeCodeHostEvent | "timeout" | "abort">((resolve) => {
			if (signal?.aborted) return resolve("abort");
			const event = this.events.shift();
			if (event) return resolve(event);
			const done = (value: ClaudeCodeHostEvent | "timeout" | "abort") => {
				clearTimeout(timer);
				signal?.removeEventListener("abort", onAbort);
				this.wake = undefined;
				resolve(value);
			};
			const onAbort = () => done("abort");
			const timer =
				ms === undefined ? undefined : setTimeout(() => done("timeout"), ms);
			signal?.addEventListener("abort", onAbort, { once: true });
			this.wake = () => {
				const event = this.events.shift();
				if (event) done(event);
			};
		});
	}

	/** Closes the session's run; it can still be resumed from where it was saved. */
	async close() {
		clearTimeout(this.timer);
		if (this.ended) return;
		this.ended = true;
		this.hooks.unregister(this);
		if (this.run) await this.host.close(this.run).catch(() => {});
	}

	/** Stops one subagent, by the call that started it. */
	async stopTask(toolCallId: string) {
		if (this.run) await this.host.stopTask(this.run, toolCallId);
	}

	private metadata(extra?: Record<string, unknown>): SharedV4ProviderMetadata {
		if (this.cursor) this.stepCursors.add(this.cursor);
		return {
			[METADATA_KEY]: {
				...(this.sessionId && { sessionId: this.sessionId }),
				...(this.cursor && { cursor: this.cursor }),
				...((extra ?? {}) as JSONObject),
			},
		};
	}

	private register(id: string, name: string, input: unknown, parent?: string) {
		const existing = this.calls.get(id);
		if (existing) {
			if (input !== undefined) existing.input = input;
			return existing;
		}
		const prefix = `mcp__${this.ctx.mcpServerName}__`;
		const kind: Call["kind"] = name.startsWith(prefix)
			? "client"
			: name === ASK_USER_QUESTION && this.ctx.questionKey && !parent
				? "question"
				: "builtin";
		const call: Call = {
			id,
			name,
			toolName:
				kind === "client"
					? name.slice(prefix.length)
					: kind === "question"
						? (this.ctx.questionKey as string)
						: (this.ctx.declared.get(name) ?? name),
			kind,
			dynamic: kind === "builtin" && !this.ctx.declared.has(name),
			input,
			parent,
			state: "open",
		};
		this.calls.set(id, call);
		return call;
	}

	private callPart(call: Call): LanguageModelV4StreamPart {
		call.announced = this.step;
		return {
			type: "tool-call",
			toolCallId: call.id,
			toolName: call.toolName,
			input: JSON.stringify(call.input ?? {}),
			...(call.kind === "builtin" && { providerExecuted: true }),
			...(call.dynamic && { dynamic: true }),
			providerMetadata: this.metadata(
				call.parent ? { parentToolCallId: call.parent } : undefined,
			),
		};
	}

	/** Calls waiting on the caller, and built-ins running that aren't blocked on it. */
	private tally() {
		let waiting = 0;
		let open = 0;
		const blocked = new Set<string>();
		for (const call of this.calls.values()) {
			if (call.state === "waiting") {
				waiting++;
				if (call.parent) blocked.add(call.parent);
			} else if (call.state === "open" && !call.parent) open++;
		}
		let running = 0;
		for (const id of this.running) {
			const call = this.calls.get(id);
			if (call?.kind === "builtin" && !blocked.has(id) && call.state === "open")
				running++;
		}
		return { waiting, open, running };
	}

	/** Streams one step: what is buffered, then on until the step ends. */
	async *stream(ctx: StepContext): AsyncGenerator<LanguageModelV4StreamPart> {
		this.ctx = ctx;
		this.attached = true;
		this.stepCursors.clear();
		clearTimeout(this.timer);
		this.step++;
		const step: Step = {
			sawTurn: false,
			turnOpen: false,
			usage: {},
			blocks: new Map(),
			inputs: new Set(),
			aborted: false,
		};
		yield { type: "stream-start", warnings: ctx.warnings };
		try {
			while (!this.ended) {
				// Nothing to wait for: an idle session only goes on when sent more.
				if (
					this.idle &&
					!step.sawTurn &&
					!this.background.size &&
					!this.events.length
				)
					break;
				const { waiting, open, running } = this.tally();
				if (!step.turnOpen) {
					if (step.result) {
						if (this.owed) yield* this.awaitOwed(step);
						break;
					}
					// Everything ran: the next turn is the next step's.
					if (
						step.sawTurn &&
						step.stopReason === "tool_use" &&
						!open &&
						!waiting
					)
						break;
				}
				const settling = !step.turnOpen && waiting > 0 && running === 0;
				const event = await this.next(
					settling ? SETTLE_MS : undefined,
					ctx.abortSignal,
				);
				if (event === "abort") {
					step.aborted = true;
					break;
				}
				if (event === "timeout") break;
				// A turn after this step's own is the next step's.
				if (
					event.type === "message" &&
					event.message.type === "stream_event" &&
					!event.message.parent_tool_use_id &&
					event.message.event.type === "message_start" &&
					step.sawTurn
				) {
					this.events.unshift(event);
					break;
				}
				yield* this.handle(event, step);
			}
			yield* this.finish(step);
		} finally {
			this.attached = false;
			this.stepCursor = this.cursor;
			if (this.cursor) this.stepCursors.add(this.cursor);
			if (!this.ended) {
				this.timer = setTimeout(() => void this.close(), this.hooks.ttlMs);
				(this.timer as { unref?: () => void }).unref?.();
			}
		}
	}

	/**
	 * After a result, with a notification owed: whether Claude Code starts a
	 * query for it, which is the next step's. What comes before is this one's.
	 */
	private async *awaitOwed(
		step: Step,
	): AsyncGenerator<LanguageModelV4StreamPart> {
		const deadline = Date.now() + OWED_MS;
		while (this.owed) {
			const event = await this.next(
				Math.max(0, deadline - Date.now()),
				this.ctx.abortSignal,
			);
			if (event === "timeout" || event === "abort") break;
			const starts =
				event.type === "message" &&
				((event.message.type === "system" &&
					event.message.subtype === "init") ||
					(event.message.type === "stream_event" &&
						!event.message.parent_tool_use_id &&
						event.message.event.type === "message_start"));
			if (starts || event.type === "end") {
				this.events.unshift(event);
				step.more = event.type !== "end";
				if (step.more) this.idle = false;
				break;
			}
			yield* this.handle(event, step);
		}
		this.owed = false;
	}

	private async *handle(
		event: ClaudeCodeHostEvent,
		step: Step,
	): AsyncGenerator<LanguageModelV4StreamPart> {
		switch (event.type) {
			case "message":
				if (this.ctx.includeRawChunks)
					yield { type: "raw", rawValue: event.message };
				yield* this.read(event.message, step, event.taskOutput);
				return;
			case "tool-call": {
				const call = event.toolUseId && this.calls.get(event.toolUseId);
				// Asked after its step ended, and answered by the next call before
				// this step took the question in.
				const early = call && this.early.get(call.id);
				if (call && early) {
					this.early.delete(call.id);
					if (this.run) await this.host.answer(this.run, event.id, early);
				} else if (call) this.questions.set(call.id, event.id);
				else if (this.run)
					await this.host.answer(this.run, event.id, {
						type: "tool-result",
						content: [{ type: "text", text: "Unknown tool call." }],
						isError: true,
					});
				return;
			}
			case "permission": {
				const call = this.register(
					event.toolUseId,
					event.name,
					event.input,
					this.calls.get(event.toolUseId)?.parent,
				);
				if (call.kind === "client") {
					if (this.run)
						await this.host.answer(this.run, event.id, {
							type: "permission",
							result: { behavior: "allow" },
						});
					return;
				}
				const early = this.early.get(call.id);
				if (early) {
					this.early.delete(call.id);
					this.asks.delete(call.id);
					if (this.run) await this.host.answer(this.run, event.id, early);
					return;
				}
				this.questions.set(call.id, event.id);
				if (call.kind === "question") return;
				this.asks.set(call.id, event);
				call.input = event.input;
				call.state = "waiting";
				// Approvals go with their calls: one asked about after its step
				// is streamed again.
				if (call.announced !== this.step) yield this.callPart(call);
				yield {
					type: "tool-approval-request",
					approvalId: call.id,
					toolCallId: call.id,
					providerMetadata: this.metadata({
						...(event.title && { title: event.title }),
						...(event.displayName && { displayName: event.displayName }),
						...(event.description && { description: event.description }),
						...(event.decisionReason && {
							decisionReason: event.decisionReason,
						}),
						...(event.blockedPath && { blockedPath: event.blockedPath }),
						...(call.parent && { parentToolCallId: call.parent }),
					}),
				};
				return;
			}
			case "tool-start":
				if (!event.agentId) this.running.add(event.toolUseId);
				return;
			case "tool-end":
				this.running.delete(event.toolUseId);
				return;
			case "error":
				this.error = event.message;
				return;
			case "end":
				this.ended = true;
				clearTimeout(this.timer);
				this.hooks.unregister(this);
				return;
		}
	}

	private *read(
		message: SDKMessage,
		step: Step,
		output?: string,
	): Generator<LanguageModelV4StreamPart> {
		const parent =
			"parent_tool_use_id" in message ? message.parent_tool_use_id : null;

		if (message.type === "system") {
			if (message.subtype === "init") {
				// A query starts, taking in what was queued. One interrupted ended
				// before it, whether its result was seen or not.
				this.owed = false;
				this.interrupted = undefined;
				this.sessionId = message.session_id;
				this.hooks.register(this);
			} else if (message.subtype === "task_started") {
				if (message.tool_use_id) {
					this.tasks.set(message.task_id, message.tool_use_id);
					if (message.is_backgrounded) this.background.add(message.tool_use_id);
				}
			} else if (message.subtype === "task_updated") {
				// A foreground call sent to the background, its result a placeholder.
				const call = this.tasks.get(message.task_id);
				if (call && message.patch.is_backgrounded) this.background.add(call);
			} else if (message.subtype === "task_notification") {
				if (!message.tool_use_id) return;
				if (!this.background.delete(message.tool_use_id)) {
					// A foreground call's result says it all, and comes after this.
					if (this.calls.has(message.tool_use_id)) return;
					// A task of a run before this one, which a resumed session
					// reports stopped: its call is settled, but no turn comes of it.
					this.stale = true;
					yield {
						type: "custom",
						kind: "claude-code.task-notification",
						providerMetadata: this.metadata({
							toolCallId: message.tool_use_id,
							taskId: message.task_id,
							status: message.status,
							summary: message.summary,
							outputFile: message.output_file,
							...(output !== undefined && { output }),
						}),
					};
					return;
				}
				// Claude Code takes it in, in a turn: work is coming.
				this.idle = false;
				this.owed = true;
				yield {
					type: "custom",
					kind: "claude-code.task-notification",
					providerMetadata: this.metadata({
						toolCallId: message.tool_use_id,
						toolName: this.calls.get(message.tool_use_id)?.toolName,
						taskId: message.task_id,
						status: message.status,
						summary: message.summary,
						outputFile: message.output_file,
						...(output !== undefined && { output }),
					}),
				};
			} else if (message.subtype === "compact_boundary") {
				yield {
					type: "custom",
					kind: "claude-code.compaction",
					providerMetadata: this.metadata({
						trigger: message.compact_metadata.trigger,
						preTokens: message.compact_metadata.pre_tokens,
					}),
				};
			}
			return;
		}

		if (parent) {
			yield* this.readSubagent(parent, message);
			return;
		}

		if (message.type === "stream_event") {
			yield* this.readStream(message.event as any, step);
			return;
		}

		if (message.type === "assistant") {
			this.cursor = message.uuid;
			const streamed = this.streamed.has(message.message.id);
			for (const [index, block] of (
				message.message.content as any[]
			).entries()) {
				if (block.type === "tool_use") {
					const call = this.register(block.id, block.name, block.input ?? {});
					if (step.inputs.delete(call.id))
						yield { type: "tool-input-end", id: call.id };
					if (call.announced === undefined) {
						yield this.callPart(call);
						if (call.kind !== "builtin") call.state = "waiting";
					}
				} else if (
					!streamed &&
					(block.type === "text" || block.type === "thinking")
				) {
					// Not streamed in parts: all at once.
					const id = `${message.message.id}:${index}`;
					const kind = block.type === "text" ? "text" : "reasoning";
					const text = block.type === "text" ? block.text : block.thinking;
					if (!text) continue;
					yield { type: `${kind}-start`, id };
					yield { type: `${kind}-delta`, id, delta: text };
					yield { type: `${kind}-end`, id, providerMetadata: this.metadata() };
				}
			}
			if (message.error && !streamed) this.error ??= message.error;
			return;
		}

		if (message.type === "user") {
			if (message.uuid) this.cursor = message.uuid;
			const content = message.message.content;
			if (typeof content === "string") return;
			for (const block of content as any[]) {
				if (block.type !== "tool_result") continue;
				const call = this.calls.get(block.tool_use_id);
				if (!call) continue;
				this.running.delete(call.id);
				this.asks.delete(call.id);
				if (call.kind !== "builtin") {
					call.state = "done";
					continue;
				}
				call.state = "done";
				yield {
					type: "tool-result",
					toolCallId: call.id,
					toolName: call.toolName,
					result: json(
						// A report handed back is the subagent's result; Claude Code's
						// says only where the model reads it.
						this.handbacks.has(call.id)
							? [{ type: "text", text: this.handbacks.get(call.id) as string }]
							: toOutput(block.content),
					) as NonNullable<JSONValue>,
					...(block.is_error && { isError: true }),
					...(call.dynamic && { dynamic: true }),
					providerMetadata: this.metadata({
						...(message.tool_use_result !== undefined && {
							output: json(message.tool_use_result),
						}),
						...(this.background.has(call.id) && { background: true }),
					}),
				};
			}
			return;
		}

		if (message.type === "result") {
			// The interrupted query's end, which the aborted step said already.
			if (this.interrupted) {
				if (!this.interrupted.sent) this.idle = true;
				this.interrupted = undefined;
				return;
			}
			// The empty query a stale notification started: what was sent is
			// taken in by the next.
			if (this.stale && !message.is_error && message.num_turns === 0) {
				this.stale = false;
				return;
			}
			this.stale = false;
			step.result = message;
			this.idle = true;
		}
	}

	private *readStream(
		event: any,
		step: Step,
	): Generator<LanguageModelV4StreamPart> {
		switch (event.type) {
			case "message_start": {
				this.idle = false;
				this.stale = false;
				step.sawTurn = true;
				step.turnOpen = true;
				step.usage.start = event.message?.usage;
				const id = event.message?.id as string | undefined;
				if (id) this.streamed.add(id);
				this.messageId = id;
				yield {
					type: "response-metadata",
					id,
					modelId: event.message?.model,
					timestamp: new Date(),
				};
				return;
			}
			case "content_block_start": {
				const block = event.content_block;
				const id = `${this.messageId}:${event.index}`;
				if (block?.type === "text") {
					step.blocks.set(id, "text");
					yield { type: "text-start", id };
				} else if (block?.type === "thinking") {
					step.blocks.set(id, "reasoning");
					yield { type: "reasoning-start", id };
				} else if (block?.type === "tool_use") {
					const call = this.register(block.id, block.name, undefined);
					this.indexes.set(event.index, call.id);
					step.inputs.add(call.id);
					yield {
						type: "tool-input-start",
						id: call.id,
						toolName: call.toolName,
						...(call.kind === "builtin" && { providerExecuted: true }),
						...(call.dynamic && { dynamic: true }),
					};
				}
				return;
			}
			case "content_block_delta": {
				const id = `${this.messageId}:${event.index}`;
				const delta = event.delta;
				if (delta?.type === "text_delta" && delta.text)
					yield { type: "text-delta", id, delta: delta.text };
				else if (delta?.type === "thinking_delta" && delta.thinking)
					yield { type: "reasoning-delta", id, delta: delta.thinking };
				else if (delta?.type === "input_json_delta" && delta.partial_json) {
					const call = this.indexes.get(event.index);
					if (call && step.inputs.has(call))
						yield {
							type: "tool-input-delta",
							id: call,
							delta: delta.partial_json,
						};
				}
				return;
			}
			case "content_block_stop": {
				const id = `${this.messageId}:${event.index}`;
				const kind = step.blocks.get(id);
				if (kind) {
					step.blocks.delete(id);
					yield { type: `${kind}-end`, id, providerMetadata: this.metadata() };
				}
				return;
			}
			case "message_delta":
				step.stopReason = event.delta?.stop_reason ?? step.stopReason;
				step.usage.delta = event.usage;
				return;
			case "message_stop":
				step.turnOpen = false;
				this.indexes.clear();
				return;
		}
	}

	private messageId?: string;
	/** The calls of the turn streaming, by block index. */
	private indexes = new Map<number, string>();

	private *readSubagent(
		parent: string,
		message: SDKMessage,
	): Generator<LanguageModelV4StreamPart> {
		const messages = this.subagents.get(parent) ?? [];
		this.subagents.set(parent, messages);
		if (message.type === "assistant") {
			const content: Extract<
				ClaudeCodeSubagentMessage,
				{ role: "assistant" }
			>["content"] = [];
			for (const block of message.message.content as any[]) {
				if (block.type === "text" && block.text)
					content.push({ type: "text", text: block.text });
				else if (block.type === "thinking" && block.thinking)
					content.push({ type: "reasoning", text: block.thinking });
				else if (block.type === "tool_use") {
					const call = this.register(block.id, block.name, block.input, parent);
					if (
						block.name === "SubagentHandback" &&
						typeof block.input?.message === "string"
					)
						this.handbacks.set(parent, block.input.message);
					content.push({
						type: "tool-call",
						toolCallId: call.id,
						toolName: call.toolName,
						input: block.input,
					});
				}
			}
			if (!content.length) return;
			messages.push({ role: "assistant", content });
		} else if (message.type === "user") {
			const content = message.message.content;
			if (typeof content === "string") return;
			const results: Extract<
				ClaudeCodeSubagentMessage,
				{ role: "tool" }
			>["content"] = [];
			for (const block of content as any[]) {
				if (block.type !== "tool_result") continue;
				const call = this.calls.get(block.tool_use_id);
				const output = toOutput(block.content);
				results.push({
					type: "tool-result",
					toolCallId: block.tool_use_id,
					toolName: call?.toolName ?? "tool",
					output,
					...(block.is_error && { isError: true }),
				});
				if (call) {
					this.asks.delete(call.id);
					call.state = "done";
					// Streamed on its own when it was asked about: its result is too.
					if (call.announced !== undefined)
						yield {
							type: "tool-result",
							toolCallId: call.id,
							toolName: call.toolName,
							result: json(output) as NonNullable<JSONValue>,
							...(block.is_error && { isError: true }),
							...(call.dynamic && { dynamic: true }),
							providerMetadata: this.metadata({ parentToolCallId: parent }),
						};
				}
			}
			if (!results.length) return;
			messages.push({ role: "tool", content: results });
		} else return;
		const call = this.calls.get(parent);
		yield {
			type: "tool-result",
			toolCallId: parent,
			toolName: call?.toolName ?? "Agent",
			result: json({ type: "subagent", messages }) as NonNullable<JSONValue>,
			preliminary: true,
			...(call?.dynamic && { dynamic: true }),
		};
	}

	private *finish(step: Step): Generator<LanguageModelV4StreamPart> {
		for (const [id, kind] of step.blocks)
			yield { type: `${kind}-end`, id, providerMetadata: this.metadata() };
		for (const id of step.inputs) yield { type: "tool-input-end", id };

		if (step.aborted && this.run) {
			// A query running ends with a result of its own, once interrupted.
			if (!this.idle) this.interrupted = { sent: false };
			await_(this.host.interrupt(this.run));
			// What waited on the caller is refused: the turn is over.
			for (const [id, question] of this.questions) {
				const call = this.calls.get(id);
				if (call) call.state = "done";
				await_(
					this.host.answer(
						this.run,
						question,
						call?.kind === "client"
							? {
									type: "tool-result",
									content: [{ type: "text", text: "Interrupted." }],
									isError: true,
								}
							: {
									type: "permission",
									result: { behavior: "deny", message: "Interrupted." },
								},
					),
				);
			}
			this.questions.clear();
			this.asks.clear();
			this.early.clear();
		}

		const result = step.result;
		if (result && this.ctx.json && "structured_output" in result) {
			const id = `${this.sessionId}:structured`;
			yield { type: "text-start", id };
			yield {
				type: "text-delta",
				id,
				delta: JSON.stringify(result.structured_output),
			};
			yield { type: "text-end", id, providerMetadata: this.metadata() };
		}

		const { waiting } = this.tally();
		const awaiting: ClaudeCodeAwaiting[] = [];
		let unified: LanguageModelV4FinishReason["unified"];
		let error: string | undefined = this.error;
		if (result) {
			if (result.is_error)
				error ??=
					("errors" in result && result.errors?.join("\n")) || result.subtype;
			unified = result.is_error
				? "error"
				: (finishReasons[step.stopReason ?? result.stop_reason ?? ""] ??
					"stop");
			if (unified === "tool-calls") unified = "stop";
			if (this.background.size || step.more) awaiting.push("background");
		} else if (step.aborted) unified = "other";
		else if (this.ended) {
			unified = "error";
			error ??= "Claude Code ended unexpectedly.";
		} else if (this.idle && !waiting) {
			// Done with its turns, with nothing to go on with.
			unified = "stop";
			if (this.background.size) awaiting.push("background");
		} else {
			unified = "tool-calls";
			const kinds = new Set(
				[...this.calls.values()]
					.filter((call) => call.state === "waiting")
					.map((call) => call.kind),
			);
			if (kinds.has("client") || kinds.has("question")) awaiting.push("tools");
			if (kinds.has("builtin")) awaiting.push("approval");
			if (!waiting) awaiting.push("continue");
		}
		if (error) {
			this.error = undefined;
			yield {
				type: "error",
				error: new ClaudeCodeError(error, result?.subtype),
			};
		}

		if (this.sessionId)
			yield {
				type: "custom",
				kind: "claude-code.step",
				providerMetadata: this.metadata({ awaiting }),
			};
		yield {
			type: "finish",
			finishReason: {
				unified,
				raw: step.stopReason ?? result?.subtype,
			},
			usage: usageOf(step.usage),
			providerMetadata: this.metadata({
				awaiting,
				...(this.ctx.route && { route: this.ctx.route }),
				...(this.background.size && { background: [...this.background] }),
				...(result && {
					costUsd: result.total_cost_usd,
					numTurns: result.num_turns,
					sessionUsage: json(result.usage),
				}),
			}),
		};
	}
}

/** Lets a promise run on its own, ignoring how it ends. */
const await_ = (promise: Promise<unknown>) => void promise.catch(() => {});
