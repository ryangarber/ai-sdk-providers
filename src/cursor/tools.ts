import {
	createProviderExecutedToolFactory,
	jsonSchema,
} from "@ai-sdk/provider-utils";
import type {
	DeleteArgs,
	EditArgs,
	GlobArgs,
	GrepArgs,
	LsArgs,
	McpArgs,
	ReadArgs,
	ReadLintsArgs,
	SemSearchArgs,
	ShellArgs,
	TaskArgs,
	UpdateTodosArgs,
	WriteArgs,
} from "@cursor/sdk";

/** The prefix of this provider's tool ids: `cursor.<Cursor's name>`. */
export const TOOL_ID_PREFIX = "cursor.";

/**
 * The tool names Cursor turns a built-in on by, where its calls have
 * another: writes come with `edit`.
 */
export const TOOL_GROUPS: Record<string, string> = { write: "edit" };

/**
 * What a built-in tool returned: `{ status: "success", value }`, or
 * `{ status: "error", error }`, in the shape of the tool's own result type.
 */
export type CursorToolOutput = unknown;

const anything = <T>() =>
	jsonSchema<T>({ type: "object", additionalProperties: true });

const builtin = <INPUT>(name: string) =>
	createProviderExecutedToolFactory<INPUT, CursorToolOutput, object>({
		id: `${TOOL_ID_PREFIX}${name}`,
		inputSchema: anything<INPUT>(),
		outputSchema: jsonSchema({}),
	});

/**
 * Cursor's own tools, to declare under any key — `tools: { read:
 * cursor.tools.read({}) }` — for typed calls. Declaring one turns it on. They
 * run in Cursor, unasked: limit them with `sandbox`, `autoReview`, or a
 * project's hooks.
 */
export const cursorTools = {
	read: builtin<ReadArgs>("read"),
	/** Comes with `write`: declaring either turns both on. */
	edit: builtin<EditArgs>("edit"),
	/** Comes with `edit`: declaring either turns both on. */
	write: builtin<WriteArgs>("write"),
	delete: builtin<DeleteArgs>("delete"),
	shell: builtin<ShellArgs>("shell"),
	grep: builtin<GrepArgs>("grep"),
	glob: builtin<GlobArgs>("glob"),
	ls: builtin<LsArgs>("ls"),
	semSearch: builtin<SemSearchArgs>("semSearch"),
	readLints: builtin<ReadLintsArgs>("readLints"),
	updateTodos: builtin<UpdateTodosArgs>("updateTodos"),
	/** Subagents. */
	task: builtin<TaskArgs>("task"),
	webSearch: builtin<{ searchTerm?: string }>("webSearch"),
	webFetch: builtin<{ url?: string }>("webFetch"),
	/** Calls to MCP servers other than the call's own tools. */
	mcp: builtin<McpArgs>("mcp"),
	/** Any other of Cursor's tools, by its name. */
	builtin: <INPUT = Record<string, unknown>>(name: string) =>
		builtin<INPUT>(name)({}),
} as const;
