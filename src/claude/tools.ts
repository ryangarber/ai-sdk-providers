import {
	createProviderDefinedToolFactoryWithOutputSchema,
	createProviderExecutedToolFactory,
	jsonSchema,
} from "@ai-sdk/provider-utils";
import type {
	AgentInput,
	AskUserQuestionInput,
	BashInput,
	ExitPlanModeInput,
	FileEditInput,
	FileReadInput,
	FileWriteInput,
	GlobInput,
	GrepInput,
	NotebookEditInput,
	TodoWriteInput,
	WebFetchInput,
	WebSearchInput,
} from "@anthropic-ai/claude-agent-sdk/sdk-tools";

/** The prefix of this provider's tool ids: `claude-code.<Claude Code's name>`. */
export const TOOL_ID_PREFIX = "claude-code.";

/** The name of Claude Code's tool for asking the user questions. */
export const ASK_USER_QUESTION = "AskUserQuestion";

/** A block of a built-in tool's result. */
export type ClaudeCodeResultBlock =
	| { type: "text"; text: string }
	| { type: "image"; mediaType: string; data: string }
	| { type: "document"; mediaType: string; data: string; title?: string };

/**
 * What a built-in tool returned, as Claude read it. Its structured output, in
 * the shape of the tool's own `*Output` type, is in the result's
 * `providerMetadata.claudeCode.output`.
 */
export type ClaudeCodeToolOutput = string | ClaudeCodeResultBlock[];

/** What a subagent has done so far, as an `Agent` call's preliminary results. */
export interface ClaudeCodeSubagentProgress {
	type: "subagent";
	messages: ClaudeCodeSubagentMessage[];
}

export type ClaudeCodeSubagentMessage =
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
				output: ClaudeCodeToolOutput;
				isError?: boolean;
			}[];
	  };

const anything = <T>() =>
	jsonSchema<T>({ type: "object", additionalProperties: true });

const builtin = <INPUT>(name: string) =>
	createProviderExecutedToolFactory<
		INPUT,
		ClaudeCodeToolOutput | ClaudeCodeSubagentProgress,
		object
	>({
		id: `${TOOL_ID_PREFIX}${name}`,
		inputSchema: anything<INPUT>(),
		outputSchema: jsonSchema({}),
		// Calls can wait on others, and so come back in a later step.
		supportsDeferredResults: true,
	});

const askUserQuestion = createProviderDefinedToolFactoryWithOutputSchema<
	AskUserQuestionInput,
	Pick<AskUserQuestionInput, "answers" | "annotations">,
	object
>({
	id: `${TOOL_ID_PREFIX}${ASK_USER_QUESTION}`,
	inputSchema: anything<AskUserQuestionInput>(),
	outputSchema:
		anything<Pick<AskUserQuestionInput, "answers" | "annotations">>(),
});

/**
 * Claude Code's own tools, to declare under their Claude Code names (any key
 * works) — `tools: { Bash: claudeCode.tools.bash({}) }` — for typed calls.
 * Declaring one turns it on. They run in Claude Code, through its permission
 * flow: a call that needs approval comes with a `tool-approval-request`.
 *
 * `askUserQuestion` is the exception: its calls are the caller's to answer,
 * with `{ answers: { [question]: answer } }` as the tool's output. Without it,
 * Claude Code can't ask.
 */
export const claudeCodeTools = {
	agent: builtin<AgentInput>("Agent"),
	bash: builtin<BashInput>("Bash"),
	edit: builtin<FileEditInput>("Edit"),
	read: builtin<FileReadInput>("Read"),
	write: builtin<FileWriteInput>("Write"),
	glob: builtin<GlobInput>("Glob"),
	grep: builtin<GrepInput>("Grep"),
	notebookEdit: builtin<NotebookEditInput>("NotebookEdit"),
	todoWrite: builtin<TodoWriteInput>("TodoWrite"),
	webFetch: builtin<WebFetchInput>("WebFetch"),
	webSearch: builtin<WebSearchInput>("WebSearch"),
	exitPlanMode: builtin<ExitPlanModeInput>("ExitPlanMode"),
	enterPlanMode: builtin<Record<string, never>>("EnterPlanMode"),
	/** Any other of Claude Code's tools, by its name. */
	builtin: <INPUT = Record<string, unknown>>(name: string) =>
		builtin<INPUT>(name)({}),
	askUserQuestion,
} as const;
