import type {
	LanguageModelV4FilePart,
	LanguageModelV4Prompt,
} from "@ai-sdk/provider";
import { z } from "zod";
import type {
	CursorAgentDefinition,
	CursorHost,
	CursorMcpServer,
	CursorModelParameter,
	CursorSettingSource,
	CursorUserImage,
} from "./host/protocol.ts";

/** A Cursor model, such as `composer-2.5`, `auto-smart` or `claude-opus-5-5`. */
export type CursorModelId = "auto-smart" | "composer-2.5" | (string & {});

const json = <T>() => z.custom<T>((value) => value !== undefined);

/**
 * `providerOptions.cursor`, per call, over the model's defaults. Options that
 * Cursor only takes when an agent is opened (instructions, built-in tools,
 * settings) reopen a live agent whose values differ, between its runs.
 */
export const CursorOptions = z.object({
	/**
	 * Cursor's own system prompt (the default), which the call's system
	 * messages go with, at the top of the agent's first message. `null`
	 * makes the call's system messages the whole of it, which Cursor only
	 * allows accounts it has enabled custom system prompts for.
	 */
	systemPromptPreset: z.enum(["cursor"]).nullish(),
	/**
	 * Cursor's own tools: `cursor` for its default set, or their names
	 * (`read`, `edit`, `shell`, …). None by default; provider-defined tools
	 * in the call add theirs.
	 */
	builtinTools: z.union([z.literal("cursor"), z.array(z.string())]).optional(),
	/** Built-in tools to take away, whatever else turns them on. */
	disallowedTools: z.array(z.string()).optional(),
	/**
	 * Which of the user's settings apply: `project` (`AGENTS.md`, rules,
	 * skills, MCP servers and hooks under `.cursor/`), `user`, `team`, `mdm`,
	 * `plugins`, or `all`. None by default.
	 */
	settingSources: json<CursorSettingSource[]>().optional(),
	/** MCP servers beyond the call's tools. */
	mcpServers: json<Record<string, CursorMcpServer>>().optional(),
	/** Subagents its `task` tool can start, by name. */
	agents: json<Record<string, CursorAgentDefinition>>().optional(),
	/** `agent` (the default) or `plan`, which only reads and proposes. */
	mode: z.enum(["agent", "plan"]).optional(),
	/**
	 * Model parameters, as `Cursor.models.list()` offers them (`fast`,
	 * `context`, `thinking`, …). Over what the call's `reasoning` picks.
	 */
	params: json<CursorModelParameter[]>().optional(),
	/** Overrides the call's `reasoning`, as the model's own value. */
	effort: z.string().optional(),
	/** Runs its shell in a sandbox: writes in `cwd`, no network. */
	sandbox: z.boolean().optional(),
	/** Has Cursor's classifier allow or block its tool calls. */
	autoReview: z.boolean().optional(),
	/** Where it works; the host's home directory if not given. */
	cwd: z.string().optional(),
	/** More directories it may work in, such as a project's other folders. */
	additionalDirectories: z.array(z.string()).optional(),
	/** A name for new agents, as Cursor lists them. */
	name: z.string().optional(),
});
export type CursorOptions = z.infer<typeof CursorOptions>;

/** A file Cursor doesn't take as it is, as text or images for it. */
export type CursorConvertFile = (
	file: LanguageModelV4FilePart,
) =>
	| string
	| CursorUserImage[]
	| undefined
	| Promise<string | CursorUserImage[] | undefined>;

export interface CursorProviderSettings {
	/**
	 * How to reach Cursor: a host, or a function that makes one on first use.
	 * Defaults to running the Cursor SDK in this process (Node and Bun), on
	 * `apiKey`. Across a process or thread boundary, pass
	 * `connectCursorHost(channel)`.
	 */
	host?: CursorHost | (() => CursorHost | Promise<CursorHost>);
	/**
	 * The API key the default host uses: one minted by `Cursor.auth.login()`,
	 * or a user or service account key. Defaults to `CURSOR_API_KEY`.
	 */
	apiKey?: string | (() => string | undefined | Promise<string | undefined>);
	/** Options for every call, under each model's and each call's own. */
	defaults?: CursorOptions;
	/**
	 * How long an agent waits, unused, for the next step before it is
	 * closed. It can still be resumed from where it was saved after. Default
	 * 10 minutes.
	 */
	sessionTtlMs?: number;
	/** Files Cursor doesn't take as they are (anything but images and text). */
	convertFile?: CursorConvertFile;
	/**
	 * The conversation before an agent, for one that can't be resumed, as
	 * text the model reads. Defaults to a plain transcript.
	 */
	transcript?: (messages: LanguageModelV4Prompt) => string | undefined;
}
