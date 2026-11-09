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
	CodexCommandDecision,
	CodexDynamicToolSpec,
	CodexResponseItem,
	CodexThreadItem,
	CodexTokenUsage,
	CodexTurn,
	CodexTurnStartParams,
	CodexUserInput,
	CodexUserInputQuestion,
} from "./host/app-server.ts";
import {
	CLOSED_METHOD,
	type CodexHistoryEntry,
	type CodexHost,
	type CodexRequestId,
	type CodexResponse,
	type CodexServerMessage,
} from "./host/protocol.ts";
import {
	METADATA_KEY,
	parseAnswers,
	type Tail,
	toToolContent,
} from "./prompt.ts";
import type { CodexSubagentMessage, CodexSubagentProgress } from "./tools.ts";

/** How long a step waits, once only the caller can move it on, for anything else. */
const SETTLE_MS = 50;

/** A step's end, as {@link LanguageModelV4StreamPart} metadata says what it waits on. */
export type CodexAwaiting =
	/** The caller's tools, and questions, to answer. */
	| "tools"
	/** Approvals of Codex's own calls. */
	| "approval"
	/** Nothing: the turn goes on, to be streamed by the next call. */
	| "continue"
	/** Subagents, which go on after the turn ended. */
	| "background";

/** A stream's `error`, from Codex. */
export class CodexError extends Error {
	override name = "CodexError";
	/** Codex's own account of the error, if it gave one. */
	readonly info?: unknown;
	constructor(message: string, info?: unknown) {
		super(message);
		this.info = info;
	}
}

/** What a step needs of the call it streams for. */
export interface StepContext {
	/** Codex's names of the built-ins the call declared, to their keys. */
	declared: Map<string, string>;
	/** The call's key for `request_user_input`, if it declared it. */
	questionKey?: string;
	tools: CodexDynamicToolSpec[];
	model?: string;
	/** What the next turn starts with, and what changes in a running one. */
	turn: Omit<CodexTurnStartParams, "threadId" | "input">;
	/** What the thread was started with, which only a new thread can change. */
	fingerprint: string;
	/** Whether `apply_patch` may write; if not, its writes are refused. */
	applyPatch: boolean;
	includeRawChunks: boolean;
	warnings: SharedV4Warning[];
	abortSignal?: AbortSignal;
	toInput: (messages: Tail["messages"]) => Promise<CodexUserInput[]>;
	/** How the step's call found its thread. */
	route?: "continue" | "fork" | "resume" | "rebuild" | "new";
}

interface Call {
	id: string;
	/** What Codex calls the tool. */
	name: string;
	/** What the call's caller calls it. */
	toolName: string;
	kind: "client" | "question" | "builtin";
	/** Declared by no call: a dynamic tool. */
	dynamic: boolean;
	input: unknown;
	/** The subagent's spawn call it was made under, if any. */
	parent?: string;
	/**
	 * `open` while Codex has it; `waiting` on the caller, for a result or an
	 * approval; `done` once answered or its result was streamed.
	 */
	state: "open" | "waiting" | "done";
	/** Where its result comes from: its item, or the raw output of its call. */
	result: "item" | "raw";
	/** The step its call was last streamed in. */
	announced?: number;
}

/** A request of Codex's waiting on the caller, by the call it is about. */
interface Pending {
	requestId: CodexRequestId;
	method: string;
	params: any;
}

/** A subagent's thread, and the call that spawned it. */
interface Child {
	threadId: string;
	spawnCall: string;
	agentPath?: string;
	status: CodexSubagentProgress["status"];
	messages: CodexSubagentMessage[];
}

/** A step's own state. */
interface Step {
	/** Its own Responses API call started. */
	sawResponse: boolean;
	responseOpen: boolean;
	/** Calls its response made. */
	madeCalls: number;
	usage?: CodexTokenUsage;
	turn?: CodexTurn;
	/** Text and reasoning blocks streaming, by id. */
	blocks: Map<string, "text" | "reasoning">;
	/**
	 * Blocks done while their response streams, ended once it has: so their
	 * metadata names the response they're in as the cursor.
	 */
	ending: Map<
		string,
		{ kind: "text" | "reasoning"; extra?: Record<string, unknown> }
	>;
	aborted: boolean;
}

/** Raw items the model makes, which mark a response streaming. */
const MODEL_ITEMS = new Set([
	"message",
	"reasoning",
	"function_call",
	"custom_tool_call",
	"local_shell_call",
	"web_search_call",
	"image_generation_call",
	"tool_search_call",
]);
/** Raw items that are calls Codex runs, and their outputs. */
const CALL_ITEMS = new Set([
	"function_call",
	"custom_tool_call",
	"local_shell_call",
	"tool_search_call",
]);
const OUTPUT_ITEMS = new Set([
	"function_call_output",
	"custom_tool_call_output",
	"local_shell_call_output",
	"tool_search_output",
]);
/** Thread items the model makes in its response, not by running a tool. */
const RESPONSE_THREAD_ITEMS = new Set([
	"agentMessage",
	"reasoning",
	"plan",
	"webSearch",
	"imageGeneration",
]);

const COLLAB_NAMES: Record<string, string> = {
	spawnAgent: "spawn_agent",
	sendInput: "send_input",
	resumeAgent: "resume_agent",
	wait: "wait_agent",
	closeAgent: "close_agent",
	sendMessage: "send_message",
	followupTask: "followup_task",
	interruptAgent: "interrupt_agent",
	listAgents: "list_agents",
};

const usageOf = (usage: CodexTokenUsage | undefined): LanguageModelV4Usage => ({
	inputTokens: {
		total: usage?.inputTokens,
		noCache:
			usage === undefined
				? undefined
				: usage.inputTokens - usage.cachedInputTokens,
		cacheRead: usage?.cachedInputTokens,
		cacheWrite: usage?.cacheWriteInputTokens,
	},
	outputTokens: {
		total: usage?.outputTokens,
		text:
			usage === undefined
				? undefined
				: usage.outputTokens - usage.reasoningOutputTokens,
		reasoning: usage?.reasoningOutputTokens,
	},
	raw: usage as unknown as JSONObject | undefined,
});

const json = (value: unknown): JSONValue =>
	value === undefined ? null : JSON.parse(JSON.stringify(value));

const parseArguments = (value: unknown) => {
	if (typeof value !== "string") return value;
	try {
		return JSON.parse(value);
	} catch {
		return value;
	}
};

/** The text of a raw call's output, as the model read it. */
const outputOf = (item: { output?: unknown }): string => {
	const output = item.output;
	if (typeof output === "string") return output;
	if (Array.isArray(output))
		return output
			.map((part: any) => (typeof part?.text === "string" ? part.text : ""))
			.join("");
	if (output && typeof output === "object" && "content" in output)
		return outputOf({ output: (output as { content: unknown }).content });
	return output === undefined ? "" : JSON.stringify(output);
};

/** A thread item a tool made, as a call: what Codex calls it, and its input. */
const describe = (
	item: CodexThreadItem,
): { name: string; input: unknown } | undefined => {
	switch (item.type) {
		case "commandExecution":
			return {
				name: "exec_command",
				input: {
					command: item.command,
					cwd: item.cwd,
					...(item.commandActions?.length && { actions: item.commandActions }),
				},
			};
		case "fileChange":
			return { name: "apply_patch", input: { changes: item.changes } };
		case "mcpToolCall":
			return {
				name: `mcp__${item.server}__${item.tool}`,
				input: item.arguments ?? {},
			};
		case "dynamicToolCall":
			return { name: item.tool, input: item.arguments ?? {} };
		case "webSearch":
			return {
				name: "web_search",
				input: {
					query: item.query,
					...(item.action ? { action: item.action } : {}),
				},
			};
		case "imageView":
			return { name: "view_image", input: { path: item.path } };
		case "imageGeneration":
			return {
				name: "image_generation",
				input: { prompt: item.revised_prompt ?? (item as any).revisedPrompt },
			};
		case "collabAgentToolCall":
			return {
				name: COLLAB_NAMES[item.tool] ?? item.tool,
				input: {
					receiverThreadIds: item.receiverThreadIds,
					...(item.prompt !== null && { prompt: item.prompt }),
					...(item.model !== null && { model: item.model }),
					...(item.reasoningEffort !== null && {
						reasoningEffort: item.reasoningEffort,
					}),
				},
			};
		case "sleep":
			return { name: "sleep", input: {} };
		default:
			return undefined;
	}
};

/** A tool's finished item, as its result, and whether it failed. */
const resultOf = (
	item: CodexThreadItem,
): { result: unknown; isError: boolean } => {
	switch (item.type) {
		case "commandExecution":
			return {
				result: {
					output: item.aggregatedOutput ?? "",
					exitCode: item.exitCode,
					status: item.status,
					durationMs: item.durationMs,
				},
				isError: item.status !== "completed",
			};
		case "fileChange":
			return {
				result: { status: item.status, changes: item.changes },
				isError: item.status !== "completed",
			};
		case "mcpToolCall":
			return item.error
				? { result: { error: item.error.message }, isError: true }
				: {
						result: {
							content: item.result?.content ?? [],
							...(item.result?.structuredContent !== undefined && {
								structuredContent: item.result.structuredContent,
							}),
						},
						isError: item.status === "failed",
					};
		case "dynamicToolCall":
			return {
				result: item.contentItems ?? [],
				isError: item.success === false,
			};
		case "webSearch":
			return {
				result: {
					query: item.query,
					...(item.action ? { action: item.action } : {}),
					...((item as any).results ? { results: (item as any).results } : {}),
				},
				isError: false,
			};
		case "imageView":
			return { result: { path: item.path }, isError: false };
		case "imageGeneration": {
			const { id: _, type: __, ...rest } = item;
			return { result: rest, isError: item.status === "failed" };
		}
		case "collabAgentToolCall":
			return {
				result: {
					status: item.status,
					receiverThreadIds: item.receiverThreadIds,
					agentsStates: item.agentsStates,
				},
				isError: item.status === "failed",
			};
		default: {
			const { id: _, type: __, ...rest } = item as any;
			return { result: rest, isError: false };
		}
	}
};

/** The id an approval streams with: its call's, and its own if it has one. */
const approvalKey = (params: { itemId: string; approvalId?: string | null }) =>
	params.approvalId ? `${params.itemId}#${params.approvalId}` : params.itemId;

/** Whether the turn failed for its context window. */
const isLength = (info: unknown) =>
	JSON.stringify(info ?? "").includes("ontextWindowExceeded");

/**
 * One Codex thread, live in a host: streamed a step at a time, and waiting
 * between steps for what the next call brings. What happens while no step
 * streams is kept for the next.
 */
export class Session {
	readonly threadId: string;
	/** What it was started with, which a continuing call must match. */
	readonly fingerprint: string;
	/** The last Responses API call seen. */
	cursor?: string;
	/** Every place the last step said it left the thread at. */
	private stepCursors = new Set<string>();
	ended = false;
	attached = false;

	/** The turn running, if any, and the last one. */
	private activeTurn?: string;
	private turnId?: string;
	private turnEnded = false;
	/** A turn has been asked for and not yet started. */
	private starting = false;
	/** Calls of the model's responses that have no output yet. */
	private openCalls = new Set<string>();
	/** What raw calls said, by call id: their tool and arguments. */
	private rawCalls = new Map<string, { name: string; input: unknown }>();
	/** The history of each turn as streamed, for rebuilding the thread. */
	private streamedHistory: CodexHistoryEntry[] = [];
	/** What the thread was given to start from, before its own turns. */
	private readonly seed: CodexResponseItem[];

	private error?: { message: string; info?: unknown };
	private events: CodexServerMessage[] = [];
	private wake?: () => void;
	private calls = new Map<string, Call>();
	/** Requests waiting on the caller, by approval or call id. */
	private pending = new Map<string, Pending>();
	/** Answers given before Codex asked. */
	private early = new Map<string, CodexResponse>();
	/** Approvals asked, by approval id, to the call they're about. */
	private approvals = new Map<string, string>();
	/** What to tell the model of denials, once the turn can take it. */
	private denials: string[] = [];
	/** Built-ins running, by call. */
	private running = new Set<string>();
	private children = new Map<string, Child>();
	/** Text and reasoning that came in parts, whose whole items add nothing. */
	private streamed = new Set<string>();
	private threadUsage?: CodexTokenUsage;
	private step = 0;
	private ctx: StepContext;
	private settings: string;
	private timer?: ReturnType<typeof setTimeout>;

	private readonly host: CodexHost;
	private readonly hooks: SessionHooks;

	constructor(
		host: CodexHost,
		threadId: string,
		ctx: StepContext,
		hooks: SessionHooks,
		seed: CodexResponseItem[] = [],
	) {
		this.host = host;
		this.threadId = threadId;
		this.seed = seed;
		this.ctx = ctx;
		this.fingerprint = ctx.fingerprint;
		this.settings = JSON.stringify(ctx.turn);
		this.hooks = hooks;
		hooks.register(this);
	}

	/** The thread's whole history, as far as this process saw it. */
	history(): CodexHistoryEntry[] {
		return [
			...this.seed.map((item) => ({ type: "item" as const, item })),
			...this.streamedHistory,
		];
	}

	/** Starts a turn with what the user said. */
	async startTurn(input: CodexUserInput[]) {
		this.starting = true;
		this.turnEnded = false;
		try {
			const { turn } = (await this.host.request("turn/start", {
				threadId: this.threadId,
				input,
				...this.ctx.turn,
			})) as { turn: CodexTurn };
			this.activeTurn ??= turn.id;
			this.turnId ??= turn.id;
		} catch (error) {
			this.starting = false;
			throw error;
		}
	}

	/** Whether a call that left the conversation at `cursor` continues here. */
	canContinue(cursor: string, fingerprint: string) {
		return this.fingerprint === fingerprint && this.isAt(cursor);
	}

	/** Whether the last step left the thread at `cursor`. */
	isAt(cursor: string) {
		return !this.ended && !this.attached && this.stepCursors.has(cursor);
	}

	/** Brings what a call changed of its settings into the live thread. */
	async update(ctx: StepContext) {
		this.ctx = ctx;
		const next = JSON.stringify(ctx.turn);
		if (next === this.settings) return;
		this.settings = next;
		// A turn that is running takes the model and effort; others go with
		// the next turn's start.
		if (!this.activeTurn) return;
		const { model, effort, summary, serviceTier, approvalPolicy } = ctx.turn;
		await this.host
			.request("turn/settings/update", {
				threadId: this.threadId,
				turnId: this.activeTurn,
				model,
				effort,
				summary,
				serviceTier,
			})
			.catch(() => {});
		if (approvalPolicy !== undefined)
			await this.host
				.request("thread/settings/update", {
					threadId: this.threadId,
					approvalPolicy,
				})
				.catch(() => {});
	}

	/**
	 * Gives the thread what came after its cursor: results, answers and
	 * approvals for what it waits on, then what the user said. Anything it
	 * waits on that the call left unanswered is refused.
	 */
	async deliver(tail: Tail) {
		for (const result of tail.results) {
			const call = this.calls.get(result.toolCallId);
			if (!call || call.state === "done") continue;
			if (call.kind === "question") {
				const questions = (
					call.input as { questions?: CodexUserInputQuestion[] }
				)?.questions;
				const answers = parseAnswers(result.output, questions ?? []);
				await this.answer(call.id, { result: { answers: answers ?? {} } });
				call.state = "done";
			} else if (call.kind === "client") {
				await this.answer(call.id, { result: toToolContent(result.output) });
				call.state = "done";
			}
		}
		for (const approval of tail.approvals) {
			const id = this.approvals.get(approval.approvalId);
			const call = id ? this.calls.get(id) : undefined;
			const pending = this.pending.get(approval.approvalId);
			if (!call || !pending) continue;
			this.approvals.delete(approval.approvalId);
			const chosen = (
				approval.providerOptions?.[METADATA_KEY] as
					| { decision?: CodexCommandDecision }
					| undefined
			)?.decision;
			await this.answer(
				approval.approvalId,
				this.decide(pending, approval.approved, chosen),
			);
			call.state = "open";
			if (!approval.approved)
				this.denials.push(
					`The user denied the ${call.name} call${approval.reason ? `: ${approval.reason}` : "."}`,
				);
		}
		// Anything still waiting is refused: the caller moved on without it.
		for (const [key, pending] of [...this.pending]) {
			const call = this.calls.get(this.approvals.get(key) ?? key);
			this.approvals.delete(key);
			await this.answer(key, this.refusal(pending));
			if (call) call.state = call.kind === "builtin" ? "open" : "done";
		}

		const input = await this.ctx.toInput(tail.messages);
		for (const denial of this.denials.splice(0))
			input.unshift({ type: "text", text: denial, text_elements: [] });
		if (!input.length) return;
		if (this.activeTurn) {
			try {
				await this.host.request("turn/steer", {
					threadId: this.threadId,
					expectedTurnId: this.activeTurn,
					input,
				});
				return;
			} catch {
				// The turn ended meanwhile: the input starts the next.
			}
		}
		await this.startTurn(input);
	}

	/** What an approval's answer is, for the kind of request it was. */
	private decide(
		pending: Pending,
		approved: boolean,
		chosen?: CodexCommandDecision,
	): CodexResponse {
		switch (pending.method) {
			case "item/permissions/requestApproval":
				return {
					result: approved
						? { permissions: pending.params.permissions, scope: "turn" }
						: { permissions: {}, scope: "turn" },
				};
			case "mcpServer/elicitation/request":
				return {
					result: {
						action: approved ? "accept" : "decline",
						content: approved ? {} : null,
						_meta: null,
					},
				};
			default:
				return {
					result: { decision: chosen ?? (approved ? "accept" : "decline") },
				};
		}
	}

	/** The answer to a request the caller didn't answer. */
	private refusal(pending: Pending): CodexResponse {
		switch (pending.method) {
			case "item/tool/call":
				return {
					result: {
						contentItems: [{ type: "inputText", text: "No result was given." }],
						success: false,
					},
				};
			case "item/tool/requestUserInput":
				return { result: { answers: {} } };
			default:
				return this.decide(pending, false);
		}
	}

	/** Answers what Codex asked about a call, or keeps it until it asks. */
	private async answer(key: string, response: CodexResponse) {
		const pending = this.pending.get(key);
		this.pending.delete(key);
		if (pending) await this.host.respond(pending.requestId, response);
		else this.early.set(key, response);
	}

	/** Takes a message from the host, answering at once what was answered early. */
	push(message: CodexServerMessage) {
		if (message.id !== undefined) {
			const key = message.params?.itemId
				? approvalKey(message.params)
				: message.params?.callId;
			const early = typeof key === "string" ? this.early.get(key) : undefined;
			if (early) {
				this.early.delete(key);
				void this.host.respond(message.id, early).catch(() => {});
				return;
			}
		}
		this.events.push(message);
		this.wake?.();
	}

	private next(ms: number | undefined, signal?: AbortSignal) {
		return new Promise<CodexServerMessage | "timeout" | "abort">((resolve) => {
			if (signal?.aborted) return resolve("abort");
			const event = this.events.shift();
			if (event) return resolve(event);
			const done = (value: CodexServerMessage | "timeout" | "abort") => {
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

	/** Unloads the thread; it can still be resumed from where it was saved. */
	async close() {
		clearTimeout(this.timer);
		if (this.ended) return;
		this.ended = true;
		this.hooks.unregister(this);
		for (const [key, pending] of [...this.pending])
			await this.answer(key, this.refusal(pending)).catch(() => {});
		if (this.activeTurn)
			await this.host
				.request("turn/interrupt", {
					threadId: this.threadId,
					turnId: this.activeTurn,
				})
				.catch(() => {});
		await this.host
			.request("thread/unsubscribe", { threadId: this.threadId })
			.catch(() => {});
	}

	/** Stops one subagent, by the call that spawned it. */
	async stopTask(toolCallId: string) {
		for (const child of this.children.values())
			if (child.spawnCall === toolCallId) {
				const { data } = (await this.host
					.request("thread/turns/list", {
						threadId: child.threadId,
						limit: 1,
					})
					.catch(() => ({ data: [] }))) as { data: CodexTurn[] };
				const turn = data.find((turn) => turn.status === "inProgress");
				if (turn)
					await this.host
						.request("turn/interrupt", {
							threadId: child.threadId,
							turnId: turn.id,
						})
						.catch(() => {});
			}
	}

	private metadata(extra?: Record<string, unknown>): SharedV4ProviderMetadata {
		if (this.cursor) this.stepCursors.add(this.cursor);
		return {
			[METADATA_KEY]: {
				threadId: this.threadId,
				...(this.turnId && { turnId: this.turnId }),
				...(this.cursor && { cursor: this.cursor }),
				fingerprint: this.fingerprint,
				...(this.ctx.model && { model: this.ctx.model }),
				...((extra ?? {}) as JSONObject),
			},
		};
	}

	private register(
		id: string,
		name: string,
		input: unknown,
		options: {
			parent?: string;
			kind?: Call["kind"];
			result?: Call["result"];
		} = {},
	) {
		const existing = this.calls.get(id);
		if (existing) {
			if (input !== undefined) existing.input = input;
			return existing;
		}
		const kind: Call["kind"] = options.kind ?? "builtin";
		const call: Call = {
			id,
			name,
			toolName:
				kind === "client"
					? name
					: kind === "question"
						? (this.ctx.questionKey as string)
						: (this.ctx.declared.get(name) ?? name),
			kind,
			dynamic: kind === "builtin" && !this.ctx.declared.has(name),
			input,
			parent: options.parent,
			state: "open",
			result: options.result ?? "item",
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

	private resultPart(
		call: Call,
		result: unknown,
		isError: boolean,
		extra?: Record<string, unknown>,
	): LanguageModelV4StreamPart {
		return {
			type: "tool-result",
			toolCallId: call.id,
			toolName: call.toolName,
			result: json(result) as NonNullable<JSONValue>,
			...(isError && { isError: true }),
			...(call.dynamic && { dynamic: true }),
			providerMetadata: this.metadata({
				...(call.parent && { parentToolCallId: call.parent }),
				...extra,
			}),
		};
	}

	/**
	 * A call's result, with its call again if `streamText` needs it in the
	 * same step: one made in the response all at once, or an undeclared one
	 * from an earlier step.
	 */
	private *finalResult(
		call: Call,
		result: unknown,
		isError: boolean,
	): Generator<LanguageModelV4StreamPart> {
		if (
			call.announced === undefined ||
			(call.dynamic && call.announced !== this.step)
		)
			yield this.callPart(call);
		yield this.resultPart(call, result, isError);
	}

	/** Calls waiting on the caller, and built-ins running that aren't blocked on it. */
	private tally() {
		const waiting = this.pending.size;
		let childWaiting = false;
		for (const pending of this.pending.values())
			if (pending.params?.threadId && pending.params.threadId !== this.threadId)
				childWaiting = true;
		let running = 0;
		for (const id of this.running) {
			const call = this.calls.get(id);
			if (call?.state !== "open") continue;
			// Waiting on subagents that wait on the caller.
			if (childWaiting && call.name === "wait_agent") continue;
			running++;
		}
		const background = [...this.children.values()].filter(
			(child) => child.status === "running",
		).length;
		return { waiting, running, background };
	}

	/** Streams one step: what is buffered, then on until the step ends. */
	async *stream(ctx: StepContext): AsyncGenerator<LanguageModelV4StreamPart> {
		this.ctx = ctx;
		this.attached = true;
		this.stepCursors.clear();
		clearTimeout(this.timer);
		this.step++;
		const step: Step = {
			sawResponse: false,
			responseOpen: false,
			madeCalls: 0,
			blocks: new Map(),
			ending: new Map(),
			aborted: false,
		};
		yield { type: "stream-start", warnings: ctx.warnings };
		try {
			while (!this.ended) {
				const { waiting, running, background } = this.tally();
				// Nothing to wait for: an idle thread only goes on when sent more.
				if (
					!this.activeTurn &&
					!this.starting &&
					!step.sawResponse &&
					!step.turn &&
					!background &&
					!this.events.length
				)
					break;
				if (!step.responseOpen) {
					if (step.turn) break;
					// Everything ran: the next response is the next step's.
					if (
						step.sawResponse &&
						step.madeCalls &&
						!this.openCalls.size &&
						!waiting
					)
						break;
				}
				const settling = !step.responseOpen && waiting > 0 && running === 0;
				const event = await this.next(
					settling ? SETTLE_MS : undefined,
					ctx.abortSignal,
				);
				if (event === "abort") {
					step.aborted = true;
					break;
				}
				if (event === "timeout") break;
				// A response after this step's own is the next step's.
				if (
					step.sawResponse &&
					!step.responseOpen &&
					this.startsResponse(event)
				) {
					this.events.unshift(event);
					break;
				}
				yield* this.handle(event, step);
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

	/** Whether a raw item is of the turn running, not one given to the thread. */
	private inTurn(params: { turnId?: string | null }) {
		return !!params.turnId && params.turnId === this.activeTurn;
	}

	/** Whether a message is the first of a Responses API call of this thread's. */
	private startsResponse(message: CodexServerMessage) {
		if (message.params?.threadId !== this.threadId) return false;
		switch (message.method) {
			case "item/started":
				return RESPONSE_THREAD_ITEMS.has(message.params.item?.type);
			case "item/agentMessage/delta":
			case "item/reasoning/summaryTextDelta":
			case "item/reasoning/textDelta":
			case "item/reasoning/summaryPartAdded":
			case "item/plan/delta":
				return true;
			case "rawResponseItem/completed":
				return (
					this.inTurn(message.params) &&
					MODEL_ITEMS.has(message.params.item?.type) &&
					message.params.item?.role !== "user" &&
					message.params.item?.role !== "developer"
				);
			default:
				return false;
		}
	}

	private *handle(
		message: CodexServerMessage,
		step: Step,
	): Generator<LanguageModelV4StreamPart> {
		if (this.ctx.includeRawChunks) yield { type: "raw", rawValue: message };
		if (message.method === CLOSED_METHOD) {
			this.ended = true;
			this.error ??= {
				message: message.params?.message ?? "Codex stopped unexpectedly.",
			};
			this.hooks.unregister(this);
			return;
		}
		const params = message.params ?? {};
		const threadId: string | undefined = params.threadId ?? params.thread?.id;
		if (threadId && threadId !== this.threadId) {
			yield* this.handleChild(threadId, message);
			return;
		}
		if (message.id !== undefined) {
			yield* this.handleRequest(message, undefined);
			return;
		}
		if (this.startsResponse(message) && !step.responseOpen) {
			step.sawResponse = true;
			step.responseOpen = true;
		}
		switch (message.method) {
			case "turn/started":
				this.starting = false;
				this.activeTurn = params.turn?.id;
				this.turnId = params.turn?.id;
				this.turnEnded = false;
				if (this.turnId)
					this.streamedHistory.push({ type: "turn", turnId: this.turnId });
				return;
			case "turn/completed": {
				const turn = params.turn as CodexTurn;
				this.activeTurn = undefined;
				this.starting = false;
				this.turnEnded = true;
				this.openCalls.clear();
				step.turn = turn;
				step.responseOpen = false;
				return;
			}
			case "rawResponseItem/completed": {
				// Items given to the thread come back too, outside of any turn.
				if (!this.inTurn(params)) return;
				const item = params.item;
				this.streamedHistory.push({
					type: "item",
					turnId: params.turnId,
					item,
				});
				if (CALL_ITEMS.has(item?.type) && typeof item.call_id === "string") {
					this.openCalls.add(item.call_id);
					step.madeCalls++;
					this.rawCalls.set(item.call_id, {
						name: item.namespace ? `${item.namespace}.${item.name}` : item.name,
						input: parseArguments(item.arguments ?? item.input),
					});
				} else if (
					OUTPUT_ITEMS.has(item?.type) &&
					typeof item.call_id === "string"
				) {
					this.openCalls.delete(item.call_id);
					const call = this.calls.get(item.call_id);
					if (call?.result === "raw" && call.state !== "done") {
						call.state = "done";
						yield* this.finalResult(call, outputOf(item), false);
					}
				}
				return;
			}
			case "rawResponse/completed":
				this.cursor = params.responseId;
				this.streamedHistory.push({
					type: "response",
					turnId: params.turnId,
					responseId: params.responseId,
				});
				if (step.responseOpen || !step.sawResponse) {
					step.sawResponse = true;
					step.usage = params.usage ?? undefined;
				}
				step.responseOpen = false;
				yield* this.flushEnding(step);
				yield {
					type: "response-metadata",
					id: params.responseId,
					...(this.ctx.model && { modelId: this.ctx.model }),
					timestamp: new Date(),
				};
				return;
			case "thread/tokenUsage/updated":
				this.threadUsage = params.tokenUsage?.total;
				return;
			case "item/agentMessage/delta":
				yield* this.textDelta(step, params.itemId, "text", params.delta);
				return;
			case "item/reasoning/summaryTextDelta":
				yield* this.textDelta(
					step,
					`${params.itemId}:s${params.summaryIndex ?? 0}`,
					"reasoning",
					params.delta,
				);
				return;
			case "item/reasoning/textDelta":
				yield* this.textDelta(
					step,
					`${params.itemId}:c${params.contentIndex ?? 0}`,
					"reasoning",
					params.delta,
				);
				return;
			case "item/started":
				yield* this.itemStarted(params.item as CodexThreadItem, undefined);
				return;
			case "item/completed":
				yield* this.itemCompleted(
					step,
					params.item as CodexThreadItem,
					undefined,
				);
				return;
			case "turn/plan/updated":
				yield {
					type: "custom",
					kind: "codex.plan",
					providerMetadata: this.metadata({
						explanation: params.explanation ?? null,
						plan: json(params.plan),
					}),
				};
				return;
			case "turn/diff/updated":
				yield {
					type: "custom",
					kind: "codex.diff",
					providerMetadata: this.metadata({ diff: params.diff }),
				};
				return;
			case "serverRequest/resolved":
				for (const [key, pending] of this.pending)
					if (pending.requestId === params.requestId) {
						this.pending.delete(key);
						const call = this.calls.get(this.approvals.get(key) ?? key);
						this.approvals.delete(key);
						if (call && call.state === "waiting")
							call.state = call.kind === "builtin" ? "open" : "done";
					}
				return;
			case "error":
				if (!params.willRetry)
					this.error = {
						message: params.error?.message ?? "Codex failed.",
						info: params.error?.codexErrorInfo,
					};
				return;
		}
	}

	private *textDelta(
		step: Step,
		id: string,
		kind: "text" | "reasoning",
		delta: string,
	): Generator<LanguageModelV4StreamPart> {
		if (!delta) return;
		if (!step.blocks.has(id)) {
			step.blocks.set(id, kind);
			this.streamed.add(id.split(":")[0]);
			yield { type: `${kind}-start`, id };
		}
		yield { type: `${kind}-delta`, id, delta };
	}

	private *endBlocks(
		step: Step,
		itemId: string,
		extra?: Record<string, unknown>,
	): Generator<LanguageModelV4StreamPart> {
		for (const [id, kind] of step.blocks)
			if (id === itemId || id.startsWith(`${itemId}:`)) {
				step.blocks.delete(id);
				yield* this.endBlock(step, id, kind, extra);
			}
	}

	/** Ends a block, or holds it until its response has ended. */
	private *endBlock(
		step: Step,
		id: string,
		kind: "text" | "reasoning",
		extra?: Record<string, unknown>,
	): Generator<LanguageModelV4StreamPart> {
		if (step.responseOpen) step.ending.set(id, { kind, extra });
		else
			yield { type: `${kind}-end`, id, providerMetadata: this.metadata(extra) };
	}

	/** Ends the blocks held for their response. */
	private *flushEnding(step: Step): Generator<LanguageModelV4StreamPart> {
		for (const [id, { kind, extra }] of step.ending)
			yield { type: `${kind}-end`, id, providerMetadata: this.metadata(extra) };
		step.ending.clear();
	}

	private *itemStarted(
		item: CodexThreadItem,
		parent: string | undefined,
	): Generator<LanguageModelV4StreamPart> {
		if (item.type === "subAgentActivity") {
			if (!parent) yield* this.subagentActivity(item);
			return;
		}
		const described = describe(item);
		if (!described) return;
		const call = this.register(item.id, described.name, described.input, {
			parent,
			kind: item.type === "dynamicToolCall" ? "client" : "builtin",
		});
		if (parent) return;
		if (call.kind === "builtin") this.running.add(call.id);
		if (call.announced === undefined) yield this.callPart(call);
		if (call.kind === "client") call.state = "waiting";
	}

	private *itemCompleted(
		step: Step,
		item: CodexThreadItem,
		parent: string | undefined,
	): Generator<LanguageModelV4StreamPart> {
		switch (item.type) {
			case "agentMessage": {
				const extra = item.phase ? { phase: item.phase } : undefined;
				if (!this.streamed.has(item.id) && item.text) {
					yield { type: "text-start", id: item.id };
					yield { type: "text-delta", id: item.id, delta: item.text };
					yield* this.endBlock(step, item.id, "text", extra);
				} else yield* this.endBlocks(step, item.id, extra);
				return;
			}
			case "reasoning": {
				if (!this.streamed.has(item.id))
					for (const [index, text] of item.summary.entries()) {
						if (!text) continue;
						const id = `${item.id}:s${index}`;
						yield { type: "reasoning-start", id };
						yield { type: "reasoning-delta", id, delta: text };
						yield* this.endBlock(step, id, "reasoning");
					}
				else yield* this.endBlocks(step, item.id);
				return;
			}
			case "contextCompaction":
				yield {
					type: "custom",
					kind: "codex.compaction",
					providerMetadata: this.metadata(),
				};
				return;
			case "subAgentActivity":
				if (!parent) yield* this.subagentActivity(item);
				return;
			case "userMessage":
			case "plan":
			case "hookPrompt":
			case "functionCallOutput":
			case "enteredReviewMode":
			case "exitedReviewMode":
				return;
		}
		const described = describe(item);
		if (!described) return;
		const call = this.register(item.id, described.name, described.input, {
			parent,
		});
		this.running.delete(call.id);
		if (call.state === "done") return;
		// The caller's own calls end with their answer; Codex's with their item.
		if (call.kind !== "builtin") {
			call.state = "done";
			return;
		}
		call.state = "done";
		for (const [approval, id] of this.approvals)
			if (id === call.id) {
				this.approvals.delete(approval);
				this.pending.delete(approval);
			}
		if (parent && call.announced === undefined) return;
		const { result, isError } = resultOf(item);
		yield* this.finalResult(call, result, isError);
	}

	/** A subagent started or finished: its spawn call, or its progress. */
	private *subagentActivity(
		item: Extract<CodexThreadItem, { type: "subAgentActivity" }>,
	): Generator<LanguageModelV4StreamPart> {
		let child = this.children.get(item.agentThreadId);
		if (item.kind === "started" && !child) {
			const raw = this.rawCalls.get(item.id);
			const call = this.register(
				item.id,
				"spawn_agent",
				{
					...(raw?.input && typeof raw.input === "object" ? raw.input : {}),
					agentPath: item.agentPath,
					agentThreadId: item.agentThreadId,
					...(item.model && { model: item.model }),
					...(item.reasoningEffort && {
						reasoningEffort: item.reasoningEffort,
					}),
				},
				{
					kind: "builtin",
					result: this.openCalls.has(item.id) ? "raw" : "item",
				},
			);
			child = {
				threadId: item.agentThreadId,
				spawnCall: call.id,
				agentPath: item.agentPath,
				status: "running",
				messages: [],
			};
			this.children.set(item.agentThreadId, child);
			this.hooks.adopt(item.agentThreadId, this);
			if (call.announced === undefined) yield this.callPart(call);
			if (call.result === "item") {
				call.state = "done";
				yield* this.finalResult(
					call,
					{ agentThreadId: item.agentThreadId, agentPath: item.agentPath },
					false,
				);
			}
			return;
		}
		if (!child) return;
		if (item.kind === "completed" || item.kind === "interrupted") {
			child.status = item.kind;
			yield this.progress(child);
		}
	}

	private progress(child: Child): LanguageModelV4StreamPart {
		const call = this.calls.get(child.spawnCall);
		const progress: CodexSubagentProgress = {
			type: "subagent",
			agentThreadId: child.threadId,
			...(child.agentPath && { agentPath: child.agentPath }),
			status: child.status,
			messages: child.messages,
		};
		return {
			type: "tool-result",
			toolCallId: child.spawnCall,
			toolName: call?.toolName ?? "spawn_agent",
			result: json(progress) as NonNullable<JSONValue>,
			preliminary: true,
			...(call?.dynamic && { dynamic: true }),
		};
	}

	/** A step for what subagents finish, which streams nothing of its own. */
	private childStep: Step = {
		sawResponse: false,
		responseOpen: false,
		madeCalls: 0,
		blocks: new Map(),
		ending: new Map(),
		aborted: false,
	};

	/** What a subagent's thread did, streamed in its spawn call. */
	private *handleChild(
		threadId: string,
		message: CodexServerMessage,
	): Generator<LanguageModelV4StreamPart> {
		const child = this.children.get(threadId);
		const parent = child?.spawnCall;
		if (message.method === "thread/started") {
			const thread = message.params.thread;
			if (thread?.parentThreadId && !this.children.has(thread.id))
				this.hooks.adopt(thread.id, this);
			return;
		}
		if (message.id !== undefined) {
			yield* this.handleRequest(message, parent);
			return;
		}
		if (!child) return;
		const params = message.params;
		if (message.method === "item/started") {
			yield* this.itemStarted(params.item, parent);
			return;
		}
		if (message.method === "turn/completed") {
			if (child.status === "running") {
				child.status =
					params.turn?.status === "interrupted" ? "interrupted" : "completed";
				yield this.progress(child);
			}
			return;
		}
		if (message.method !== "item/completed") return;
		const item = params.item as CodexThreadItem;
		if (describe(item)) yield* this.itemCompleted(this.childStep, item, parent);
		if (item.type === "agentMessage" && item.text)
			child.messages.push({
				role: "assistant",
				content: [{ type: "text", text: item.text }],
			});
		else if (item.type === "reasoning" && item.summary.join(""))
			child.messages.push({
				role: "assistant",
				content: [{ type: "reasoning", text: item.summary.join("\n\n") }],
			});
		else {
			const described = describe(item);
			if (!described) return;
			const call = this.calls.get(item.id);
			const { result, isError } = resultOf(item);
			child.messages.push(
				{
					role: "assistant",
					content: [
						{
							type: "tool-call",
							toolCallId: item.id,
							toolName: call?.toolName ?? described.name,
							input: described.input,
						},
					],
				},
				{
					role: "tool",
					content: [
						{
							type: "tool-result",
							toolCallId: item.id,
							toolName: call?.toolName ?? described.name,
							output: result,
							...(isError && { isError: true }),
						},
					],
				},
			);
		}
		yield this.progress(child);
	}

	/** Something Codex asks: the caller's tools, its questions, approvals. */
	private *handleRequest(
		message: CodexServerMessage,
		parent: string | undefined,
	): Generator<LanguageModelV4StreamPart> {
		const params = message.params ?? {};
		const requestId = message.id as CodexRequestId;
		const respond = (response: CodexResponse) =>
			void this.host.respond(requestId, response).catch(() => {});
		// Asked after its step ended, and answered by the next call before this
		// step took the request in.
		const key = params.itemId ? approvalKey(params) : params.callId;
		const early = typeof key === "string" ? this.early.get(key) : undefined;
		if (early) {
			this.early.delete(key as string);
			respond(early);
			return;
		}
		switch (message.method) {
			case "item/tool/call": {
				const known = this.ctx.tools.some((tool) => tool.name === params.tool);
				if (!known) {
					respond({
						result: {
							contentItems: [
								{ type: "inputText", text: `No tool ${params.tool}.` },
							],
							success: false,
						},
					});
					return;
				}
				const call = this.register(
					params.callId,
					params.tool,
					params.arguments ?? {},
					{
						parent,
						kind: "client",
					},
				);
				this.pending.set(call.id, {
					requestId,
					method: message.method,
					params,
				});
				if (call.announced !== this.step && call.state !== "waiting")
					yield this.callPart(call);
				call.state = "waiting";
				return;
			}
			case "item/tool/requestUserInput": {
				if (!this.ctx.questionKey || parent) {
					respond({ result: { answers: {} } });
					return;
				}
				const call = this.register(
					params.itemId,
					"request_user_input",
					{ questions: params.questions },
					{ kind: "question" },
				);
				this.pending.set(call.id, {
					requestId,
					method: message.method,
					params,
				});
				if (call.announced !== this.step) yield this.callPart(call);
				call.state = "waiting";
				return;
			}
			case "item/commandExecution/requestApproval":
			case "item/fileChange/requestApproval":
			case "item/permissions/requestApproval": {
				const itemId: string = params.itemId;
				if (
					message.method === "item/fileChange/requestApproval" &&
					!this.ctx.applyPatch
				) {
					respond({ result: { decision: "decline" } });
					return;
				}
				const approvalId = approvalKey(params);
				let call = this.calls.get(itemId);
				if (!call)
					call = this.register(
						itemId,
						message.method === "item/permissions/requestApproval"
							? "request_permissions"
							: message.method === "item/fileChange/requestApproval"
								? "apply_patch"
								: "exec_command",
						message.method === "item/permissions/requestApproval"
							? { permissions: params.permissions, reason: params.reason }
							: { command: params.command, cwd: params.cwd },
						{ parent, kind: "builtin" },
					);
				this.pending.set(approvalId, {
					requestId,
					method: message.method,
					params,
				});
				this.approvals.set(approvalId, call.id);
				call.state = "waiting";
				// Approvals go with their calls: one asked about after its step
				// is streamed again.
				if (call.announced !== this.step) yield this.callPart(call);
				yield {
					type: "tool-approval-request",
					approvalId,
					toolCallId: call.id,
					providerMetadata: this.metadata({
						...(params.reason && { reason: params.reason }),
						...(params.command && { command: params.command }),
						...(params.grantRoot && { grantRoot: params.grantRoot }),
						...(params.availableDecisions && {
							availableDecisions: json(params.availableDecisions),
						}),
						...(params.proposedExecpolicyAmendment && {
							proposedExecpolicyAmendment: json(
								params.proposedExecpolicyAmendment,
							),
						}),
						...(call.parent && { parentToolCallId: call.parent }),
					}),
				};
				return;
			}
			case "mcpServer/elicitation/request":
				// No way to ask the caller for a form: declined.
				respond({ result: { action: "decline", content: null, _meta: null } });
				return;
			default:
				respond({
					error: { code: -32601, message: `Unsupported: ${message.method}` },
				});
		}
	}

	private *finish(step: Step): Generator<LanguageModelV4StreamPart> {
		yield* this.flushEnding(step);
		for (const [id, kind] of step.blocks)
			yield { type: `${kind}-end`, id, providerMetadata: this.metadata() };

		if (step.aborted) {
			// What waited on the caller is refused: the turn is over.
			for (const [key, pending] of [...this.pending]) {
				const call = this.calls.get(this.approvals.get(key) ?? key);
				if (call) call.state = "done";
				void this.answer(key, this.refusal(pending)).catch(() => {});
			}
			this.approvals.clear();
			this.early.clear();
			if (this.activeTurn)
				void this.host
					.request("turn/interrupt", {
						threadId: this.threadId,
						turnId: this.activeTurn,
					})
					.catch(() => {});
		}

		const { waiting, background } = this.tally();
		const awaiting: CodexAwaiting[] = [];
		let unified: LanguageModelV4FinishReason["unified"];
		let error = this.error;
		const turn = step.turn;
		if (turn) {
			if (turn.status === "failed") {
				error ??= {
					message: turn.error?.message ?? "The turn failed.",
					info: turn.error?.codexErrorInfo,
				};
				unified = isLength(error.info) ? "length" : "error";
			} else if (turn.status === "interrupted") unified = "other";
			else unified = "stop";
			if (background) awaiting.push("background");
		} else if (step.aborted) unified = "other";
		else if (this.ended) {
			unified = "error";
			error ??= { message: "Codex ended unexpectedly." };
		} else if (!this.activeTurn && !this.starting && !waiting) {
			unified = "stop";
			if (background) awaiting.push("background");
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
			yield { type: "error", error: new CodexError(error.message, error.info) };
		}

		const ended = { turnEnded: this.turnEnded };
		yield {
			type: "custom",
			kind: "codex.step",
			providerMetadata: this.metadata({ awaiting, ...ended }),
		};
		yield {
			type: "finish",
			finishReason: { unified, raw: turn?.status },
			usage: usageOf(step.usage),
			providerMetadata: this.metadata({
				awaiting,
				...ended,
				...(this.ctx.route && { route: this.ctx.route }),
				...(background && {
					background: [...this.children.values()]
						.filter((child) => child.status === "running")
						.map((child) => child.spawnCall),
				}),
				...(this.threadUsage && { threadUsage: json(this.threadUsage) }),
			}),
		};
	}
}

export interface SessionHooks {
	register: (session: Session) => void;
	unregister: (session: Session) => void;
	/** Routes a subagent's thread to the session that spawned it. */
	adopt: (threadId: string, session: Session) => void;
	ttlMs: number;
}

/**
 * The threads a provider has live, by thread id, with the host's messages
 * routed to them (and their subagents' to them).
 */
export class SessionRegistry {
	private sessions = new Map<string, Session>();
	private owners = new Map<string, Session>();
	private listening?: () => void;

	/** Starts routing what `host` says to the sessions it is about. */
	attach(host: CodexHost) {
		this.listening ??= host.listen((message) => this.route(message));
	}

	private route(message: CodexServerMessage) {
		if (message.method === CLOSED_METHOD) {
			for (const session of this.all()) session.push(message);
			return;
		}
		const params = message.params ?? {};
		const threadId: string | undefined = params.threadId ?? params.thread?.id;
		if (!threadId) return;
		const session = this.sessions.get(threadId) ?? this.owners.get(threadId);
		if (session) {
			session.push(message);
			return;
		}
		// A subagent's thread, before it says whose it is.
		const parent = params.thread?.parentThreadId;
		const owner = parent
			? (this.sessions.get(parent) ?? this.owners.get(parent))
			: undefined;
		if (owner) {
			this.owners.set(threadId, owner);
			owner.push(message);
		}
	}

	get = (id: string) => this.sessions.get(id);
	hooks = (ttlMs: number): SessionHooks => ({
		register: (session) => this.sessions.set(session.threadId, session),
		unregister: (session) => {
			if (this.sessions.get(session.threadId) === session)
				this.sessions.delete(session.threadId);
			for (const [id, owner] of this.owners)
				if (owner === session) this.owners.delete(id);
		},
		adopt: (threadId, session) => this.owners.set(threadId, session),
		ttlMs,
	});
	all = () => [...this.sessions.values()];
}
