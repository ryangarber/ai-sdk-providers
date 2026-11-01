import type {
	LanguageModelV4FilePart,
	LanguageModelV4Prompt,
} from "@ai-sdk/provider";
import type {
	EffortLevel,
	PermissionMode,
	SettingSource,
	ThinkingConfig,
} from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";
import type {
	ClaudeCodeHost,
	ClaudeCodeMcpServer,
	ClaudeCodeRunOptions,
} from "./host/protocol.ts";

/** A model alias Claude Code follows to its newest model, or a full model id. */
export type ClaudeCodeModelId = "opus" | "sonnet" | "haiku" | (string & {});

const json = <T>() => z.custom<T>((value) => value !== undefined);

/**
 * `providerOptions.claudeCode`, per call, over the model's defaults. Each is
 * an Agent SDK option of the same name unless it says otherwise.
 */
export const ClaudeCodeOptions = z.object({
	/**
	 * Claude Code's own system prompt, which the call's system messages are
	 * appended to. Without it (the default), the call's system messages are
	 * the whole of it.
	 */
	systemPromptPreset: z.enum(["claude_code"]).nullish(),
	/** With the preset: leave out its per-machine sections, for caching. */
	excludeDynamicSections: z.boolean().optional(),
	/**
	 * Claude Code's own tools: `claude_code` for its default set, or their
	 * names. None by default; provider-defined tools in the call add theirs.
	 */
	builtinTools: z
		.union([z.literal("claude_code"), z.array(z.string())])
		.optional(),
	allowedTools: z.array(z.string()).optional(),
	disallowedTools: z.array(z.string()).optional(),
	/**
	 * Which of the user's settings files apply: their `CLAUDE.md`, agents,
	 * skills, MCP servers, permissions and hooks. None by default.
	 */
	settingSources: json<SettingSource[]>().optional(),
	/**
	 * Settings above all of those, as `settings.json` takes them: an object,
	 * or the path of a file holding one.
	 */
	settings: z.union([z.string(), z.record(z.string(), z.unknown())]).optional(),
	/** MCP servers beyond the call's tools, configured as JSON. */
	mcpServers: json<Record<string, ClaudeCodeMcpServer>>().optional(),
	/** Only the MCP servers given here, not the user's. */
	strictMcpConfig: z.boolean().optional(),
	permissionMode: json<PermissionMode>().optional(),
	/** Overrides the call's `reasoning`. */
	effort: json<EffortLevel>().optional(),
	thinking: json<ThinkingConfig>().optional(),
	/** One of Claude Code's agents, to run the main thread as. */
	agent: z.string().optional(),
	agents: json<ClaudeCodeRunOptions["agents"]>().optional(),
	/** One of Claude Code's output styles (a setting). */
	outputStyle: z.string().optional(),
	/** Claude Code's fast mode, where the model has it (a setting). */
	fastMode: z.boolean().optional(),
	/** The `claude` to run, in place of the one the host finds. */
	executable: z.string().optional(),
	/** Where it works; the host's home directory if not given. */
	cwd: z.string().optional(),
	/** More directories it may work in, such as a project's other folders. */
	additionalDirectories: z.array(z.string()).optional(),
	fallbackModel: z.string().optional(),
	maxTurns: z.number().optional(),
	maxBudgetUsd: z.number().optional(),
	/**
	 * Whether a session resumed from disk is forked (the default), so one
	 * session never serves two branches of a conversation.
	 */
	forkSession: z.boolean().optional(),
	/** Whether sessions are saved, so they can be resumed later (default). */
	persistSession: z.boolean().optional(),
	/** Agent SDK options not covered above, passed through as they are. */
	sdkOptions: json<Partial<ClaudeCodeRunOptions>>().optional(),
});
export type ClaudeCodeOptions = z.infer<typeof ClaudeCodeOptions>;

/** A file Claude doesn't read itself, as text or content blocks for it. */
export type ClaudeCodeConvertFile = (
	file: LanguageModelV4FilePart,
) =>
	| string
	| ClaudeCodeContentBlock[]
	| undefined
	| Promise<string | ClaudeCodeContentBlock[] | undefined>;

/** A content block of a user message, as the Messages API takes it. */
export type ClaudeCodeContentBlock =
	| { type: "text"; text: string }
	| {
			type: "image";
			source:
				| { type: "base64"; media_type: string; data: string }
				| { type: "url"; url: string };
	  }
	| {
			type: "document";
			source:
				| { type: "base64"; media_type: "application/pdf"; data: string }
				| { type: "url"; url: string }
				| { type: "text"; media_type: "text/plain"; data: string };
			title?: string;
	  };

export interface ClaudeCodeProviderSettings {
	/**
	 * How to reach Claude Code: a host, or a function that makes one on first
	 * use. Defaults to running it in this process (Node and Bun). Across a
	 * process or thread boundary, pass `connectClaudeCodeHost(channel)`.
	 */
	host?: ClaudeCodeHost | (() => ClaudeCodeHost | Promise<ClaudeCodeHost>);
	/** Options for every call, under each model's and each call's own. */
	defaults?: ClaudeCodeOptions;
	/**
	 * How long a session waits, unused, for the next step before it is
	 * closed. It can still be resumed from disk after. Default 10 minutes.
	 */
	sessionTtlMs?: number;
	/** The MCP server the call's tools are served under. Default `ai-sdk`. */
	mcpServerName?: string;
	/** Files Claude doesn't take as they are (anything but images and PDFs). */
	convertFile?: ClaudeCodeConvertFile;
	/**
	 * The conversation before a session, for one that can't be resumed, as
	 * text the model reads. Defaults to a plain transcript.
	 */
	transcript?: (messages: LanguageModelV4Prompt) => string | undefined;
}
