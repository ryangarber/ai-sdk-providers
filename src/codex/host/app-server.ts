/**
 * The parts of Codex's app-server protocol (`codex app-server`, JSON-RPC over
 * stdio, v2 with `experimentalApi`) this provider uses, from the bindings
 * `codex app-server generate-ts --experimental` makes (codex-cli 0.164).
 */

type JsonValue =
	| null
	| boolean
	| number
	| string
	| JsonValue[]
	| { [key: string]: JsonValue };

export type CodexReasoningEffort =
	| "none"
	| "minimal"
	| "low"
	| "medium"
	| "high"
	| "xhigh"
	| "max"
	| "ultra"
	| (string & {});

export type CodexReasoningSummary = "auto" | "concise" | "detailed" | "none";

export type CodexApprovalPolicy =
	| "untrusted"
	| "on-failure"
	| "on-request"
	| "never"
	| {
			granular: {
				sandbox_approval: boolean;
				rules: boolean;
				skill_approval: boolean;
				request_permissions: boolean;
				mcp_elicitations: boolean;
			};
	  };

export type CodexSandboxMode =
	| "read-only"
	| "workspace-write"
	| "danger-full-access";

export interface CodexDynamicToolSpec {
	type: "function";
	name: string;
	description: string;
	inputSchema: JsonValue;
	deferLoading?: boolean;
}

export type CodexUserInput =
	| { type: "text"; text: string; text_elements: unknown[] }
	| {
			type: "image";
			url: string;
			detail?: "auto" | "low" | "high" | "original";
	  }
	| { type: "localImage"; path: string }
	| { type: "skill"; name: string; path: string }
	| { type: "mention"; name: string; path: string };

/** How a thread works, set when it starts. */
interface CodexThreadSettings {
	model?: string | null;
	modelProvider?: string | null;
	serviceTier?: string | null;
	cwd?: string | null;
	runtimeWorkspaceRoots?: string[] | null;
	approvalPolicy?: CodexApprovalPolicy | null;
	sandbox?: CodexSandboxMode | null;
	permissions?: string | null;
	config?: Record<string, JsonValue> | null;
	baseInstructions?: string | null;
	developerInstructions?: string | null;
}

export interface CodexThreadStartParams extends CodexThreadSettings {
	ephemeral?: boolean | null;
	dynamicTools?: CodexDynamicToolSpec[] | null;
	experimentalRawEvents?: boolean;
	[key: string]: unknown;
}

export interface CodexTurnStartParams {
	threadId: string;
	input: CodexUserInput[];
	cwd?: string | null;
	approvalPolicy?: CodexApprovalPolicy | null;
	model?: string | null;
	serviceTier?: string | null;
	effort?: CodexReasoningEffort | null;
	summary?: CodexReasoningSummary | null;
	outputSchema?: JsonValue | null;
	[key: string]: unknown;
}

interface CodexThread {
	id: string;
	forkedFromId: string | null;
	parentThreadId: string | null;
	path: string | null;
	cwd: string;
	status: { type: "notLoaded" | "idle" | "systemError" | "active" };
	turns: CodexTurn[];
}

export interface CodexThreadResponse {
	thread: CodexThread;
	model: string;
	cwd: string;
}

interface CodexTurnError {
	message: string;
	codexErrorInfo: unknown;
	additionalDetails: string | null;
}

export interface CodexTurn {
	id: string;
	items: CodexThreadItem[];
	status: "completed" | "interrupted" | "failed" | "inProgress";
	error: CodexTurnError | null;
}

export interface CodexTokenUsage {
	totalTokens: number;
	inputTokens: number;
	cachedInputTokens: number;
	cacheWriteInputTokens: number;
	outputTokens: number;
	reasoningOutputTokens: number;
}

export type CodexFileChange = {
	path: string;
	kind: { type: "add" | "delete" | "update"; move_path?: string | null };
	diff: string;
};

/** What a thread holds, as app-server reports it. */
export type CodexThreadItem =
	| { type: "userMessage"; id: string; content: CodexUserInput[] }
	| {
			type: "agentMessage";
			id: string;
			text: string;
			phase: "commentary" | "partial_answer" | "final_answer" | null;
	  }
	| { type: "reasoning"; id: string; summary: string[]; content: string[] }
	| { type: "plan"; id: string; text: string }
	| {
			type: "commandExecution";
			id: string;
			command: string;
			cwd: string;
			processId: string | null;
			source: string;
			status: "inProgress" | "completed" | "failed" | "declined";
			commandActions: unknown[];
			aggregatedOutput: string | null;
			exitCode: number | null;
			durationMs: number | null;
	  }
	| {
			type: "fileChange";
			id: string;
			changes: CodexFileChange[];
			status: "inProgress" | "completed" | "failed" | "declined";
	  }
	| {
			type: "mcpToolCall";
			id: string;
			server: string;
			tool: string;
			status: "inProgress" | "completed" | "failed";
			arguments: JsonValue;
			result: { content: unknown[]; structuredContent?: unknown } | null;
			error: { message: string } | null;
			durationMs: number | null;
	  }
	| {
			type: "dynamicToolCall";
			id: string;
			namespace: string | null;
			tool: string;
			arguments: JsonValue;
			status: "inProgress" | "completed" | "failed";
			contentItems: CodexToolContent[] | null;
			success: boolean | null;
	  }
	| {
			type: "collabAgentToolCall";
			id: string;
			tool: string;
			status: "inProgress" | "completed" | "failed";
			senderThreadId: string;
			receiverThreadIds: string[];
			prompt: string | null;
			model: string | null;
			reasoningEffort: string | null;
			agentsStates: Record<string, unknown>;
	  }
	| {
			type: "subAgentActivity";
			id: string;
			kind: string;
			agentThreadId: string;
			agentPath: string;
			model: string | null;
			reasoningEffort: string | null;
	  }
	| {
			type: "webSearch";
			id: string;
			query: string;
			action?: unknown;
	  }
	| { type: "imageView"; id: string; path: string }
	| {
			type: "imageGeneration";
			id: string;
			status: string;
			revised_prompt?: string | null;
			result: string;
			[key: string]: unknown;
	  }
	| { type: "sleep"; id: string; [key: string]: unknown }
	| { type: "contextCompaction"; id: string }
	| { type: "hookPrompt"; id: string; fragments: unknown[] }
	| { type: "functionCallOutput"; id: string; name: string; output: unknown }
	| { type: "enteredReviewMode"; id: string; review: string }
	| { type: "exitedReviewMode"; id: string; review: string };

export type CodexToolContent =
	| { type: "inputText"; text: string }
	| { type: "inputImage"; imageUrl: string };

export interface CodexUserInputQuestion {
	id: string;
	header: string;
	question: string;
	isOther: boolean;
	isSecret: boolean;
	options: { label: string; description: string }[] | null;
}

export type CodexCommandDecision =
	| "accept"
	| "acceptForSession"
	| "decline"
	| "cancel"
	| Record<string, unknown>;

/** A Responses API item, as Codex keeps it. */
export interface CodexResponseItem {
	type: string;
	id?: string;
	role?: string;
	call_id?: string;
	name?: string;
	namespace?: string;
	content?: { type: string; text?: string }[];
	[key: string]: unknown;
}

export interface CodexModel {
	id: string;
	model: string;
	displayName: string;
	description: string;
	hidden: boolean;
	isDefault: boolean;
	defaultReasoningEffort: CodexReasoningEffort;
	supportedReasoningEfforts: { reasoningEffort: CodexReasoningEffort }[];
	inputModalities: string[];
	serviceTiers?: { id: string; name?: string }[];
}
