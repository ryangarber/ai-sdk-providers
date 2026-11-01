import {
	createProviderDefinedToolFactoryWithOutputSchema,
	createProviderExecutedToolFactory,
	jsonSchema,
} from "@ai-sdk/provider-utils";
import type {
	CodexFileChange,
	CodexUserInputQuestion,
} from "./host/app-server.ts";

/** The prefix of this provider's tool ids: `codex.<Codex's name>`. */
export const TOOL_ID_PREFIX = "codex.";

/** Codex's tool for asking the user questions. */
export const REQUEST_USER_INPUT = "request_user_input";

/** Codex's own tools, by the names their calls stream under. */
export const BUILTIN_NAMES = [
	"exec_command",
	"apply_patch",
	"web_search",
	"view_image",
	"image_generation",
	"update_plan",
	"spawn_agent",
] as const;
export type CodexBuiltinName = (typeof BUILTIN_NAMES)[number];

export interface CodexExecCommandInput {
	command: string;
	cwd: string;
	/** Codex's best guess at what the command does, for display. */
	actions?: unknown[];
}
export interface CodexExecCommandOutput {
	output: string;
	exitCode: number | null;
	status: "completed" | "failed" | "declined";
	durationMs?: number | null;
}
export interface CodexApplyPatchInput {
	changes: CodexFileChange[];
}
export interface CodexApplyPatchOutput {
	status: "completed" | "failed" | "declined";
	changes: CodexFileChange[];
}
export interface CodexWebSearchInput {
	query: string;
	action?: unknown;
}
export interface CodexViewImageInput {
	path: string;
}
export interface CodexImageGenerationInput {
	prompt?: string;
}
export interface CodexImageGenerationOutput {
	status: string;
	/** The image, base64, when it came inline. */
	result?: string;
	revisedPrompt?: string | null;
	savedPath?: string;
}
export interface CodexSpawnAgentInput {
	prompt?: string | null;
	model?: string | null;
	reasoningEffort?: string | null;
	agentPath?: string;
	agentThreadId?: string;
}

/** What a subagent has done so far, as its spawn call's preliminary results. */
export interface CodexSubagentProgress {
	type: "subagent";
	agentThreadId: string;
	agentPath?: string;
	status: "running" | "completed" | "interrupted";
	messages: CodexSubagentMessage[];
}

export type CodexSubagentMessage =
	| {
			role: "assistant";
			content: (
				| { type: "text"; text: string }
				| { type: "reasoning"; text: string }
				| {
						type: "tool-call";
						toolCallId: string;
						toolName: string;
						input: unknown;
				  }
			)[];
	  }
	| {
			role: "tool";
			content: {
				type: "tool-result";
				toolCallId: string;
				toolName: string;
				output: unknown;
				isError?: boolean;
			}[];
	  };

export interface CodexRequestUserInputInput {
	questions: CodexUserInputQuestion[];
}
/**
 * The answers to a `request_user_input` call: by question id (or the
 * question's text), one answer or several.
 */
export interface CodexRequestUserInputOutput {
	answers: Record<string, string | string[]>;
}

const anything = <T>() =>
	jsonSchema<T>({ type: "object", additionalProperties: true });

const builtin = <INPUT, OUTPUT = unknown>(name: string) =>
	createProviderExecutedToolFactory<
		INPUT,
		OUTPUT | CodexSubagentProgress,
		object
	>({
		id: `${TOOL_ID_PREFIX}${name}`,
		inputSchema: anything<INPUT>(),
		outputSchema: jsonSchema({}),
		// Calls can wait on others, and so come back in a later step.
		supportsDeferredResults: true,
	});

const requestUserInput = createProviderDefinedToolFactoryWithOutputSchema<
	CodexRequestUserInputInput,
	CodexRequestUserInputOutput,
	object
>({
	id: `${TOOL_ID_PREFIX}${REQUEST_USER_INPUT}`,
	inputSchema: anything<CodexRequestUserInputInput>(),
	outputSchema: anything<CodexRequestUserInputOutput>(),
});

/**
 * Codex's own tools, to declare under any key — `tools: { shell:
 * codex.tools.execCommand({}) }` — for typed calls. Declaring one turns it
 * on. They run in Codex, inside its sandbox: a call that needs approval comes
 * with a `tool-approval-request`.
 *
 * `requestUserInput` is the exception: its calls are the caller's to answer,
 * with `{ answers: { [questionId]: answer } }` as the tool's output. Without
 * it, Codex can't ask.
 */
export const codexTools = {
	execCommand: builtin<CodexExecCommandInput, CodexExecCommandOutput>(
		"exec_command",
	),
	applyPatch: builtin<CodexApplyPatchInput, CodexApplyPatchOutput>(
		"apply_patch",
	),
	webSearch: builtin<CodexWebSearchInput>("web_search"),
	viewImage: builtin<CodexViewImageInput>("view_image"),
	imageGeneration: builtin<
		CodexImageGenerationInput,
		CodexImageGenerationOutput
	>("image_generation"),
	/** Its plan, as `codex.plan` custom parts. */
	updatePlan: builtin<Record<string, unknown>>("update_plan"),
	/** Subagents: `spawn_agent` and the tools that work with them. */
	spawnAgent: builtin<CodexSpawnAgentInput>("spawn_agent"),
	/** Any other of Codex's tools, by its name. */
	builtin: <INPUT = Record<string, unknown>>(name: string) =>
		builtin<INPUT>(name)({}),
	requestUserInput,
} as const;
