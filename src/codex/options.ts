import type {
	LanguageModelV4FilePart,
	LanguageModelV4Prompt,
} from "@ai-sdk/provider";
import { z } from "zod";
import type {
	CodexApprovalPolicy,
	CodexReasoningEffort,
	CodexReasoningSummary,
	CodexSandboxMode,
	CodexThreadStartParams,
	CodexUserInput,
} from "./host/app-server.ts";
import type { CodexHost } from "./host/protocol.ts";

/** A Codex model, such as `gpt-6.1-sol`, or `default` for the account's. */
export type CodexModelId = "default" | (string & {});

const json = <T>() => z.custom<T>((value) => value !== undefined);

/**
 * `providerOptions.codex`, per call, over the model's defaults. Options that
 * Codex only takes when a thread starts (instructions, tools, config) make a
 * call whose values differ go on in a new thread rebuilt from the old one.
 */
export const CodexOptions = z.object({
	/**
	 * Codex's own instructions, which the call's system messages go after as
	 * developer instructions. Without it (the default), the call's system
	 * messages are the whole of them.
	 */
	systemPromptPreset: z.enum(["codex"]).nullish(),
	/**
	 * Codex's own tools: `codex` for whatever its config turns on, or their
	 * names (`exec_command`, `web_search`, …). None by default;
	 * provider-defined tools in the call add theirs. `apply_patch` comes
	 * with most models regardless: without it, writes are refused.
	 */
	builtinTools: z.union([z.literal("codex"), z.array(z.string())]).optional(),
	/**
	 * Whether the user's own Codex setup applies: `AGENTS.md` files, skills,
	 * memories, plugins, apps, hooks and MCP servers. Off by default. Its
	 * `config.toml` always picks the model provider and sign-in.
	 */
	userConfig: z.boolean().optional(),
	/** When Codex asks before it acts. */
	approvalPolicy: json<CodexApprovalPolicy>().optional(),
	/**
	 * What its commands may touch. `read-only` by default when `apply_patch`
	 * isn't one of its tools, so writes have to ask (and are refused).
	 */
	sandbox: json<CodexSandboxMode>().optional(),
	/** A named permissions profile, in place of `sandbox`. */
	permissions: z.string().optional(),
	/** Overrides the call's `reasoning`. */
	effort: json<CodexReasoningEffort>().optional(),
	/** How much of its reasoning Codex summarizes, as reasoning parts. */
	reasoningSummary: json<CodexReasoningSummary>().optional(),
	/** A service tier, such as fast mode's, where the model offers it. */
	serviceTier: z.string().optional(),
	/** Where it works; the host's home directory if not given. */
	cwd: z.string().optional(),
	/** More directories it may work in, such as a project's other folders. */
	additionalDirectories: z.array(z.string()).optional(),
	/** Config overrides, as dotted keys or tables (`-c key=value`). */
	config: z.record(z.string(), z.unknown()).optional(),
	/** MCP servers beyond the call's tools, as `config.toml` takes them. */
	mcpServers: z.record(z.string(), z.unknown()).optional(),
	/** Whether threads are saved, so they can be resumed later (default). */
	persistThread: z.boolean().optional(),
	/**
	 * How long a script of Codex's (its `exec` tool, in code mode) runs before
	 * handing back to the model. Long by default, so a call waiting on the
	 * caller doesn't spend turns polling. Default an hour.
	 */
	execYieldMs: z.number().optional(),
	/** `thread/start` params not covered above, passed through as they are. */
	threadOptions: json<Partial<CodexThreadStartParams>>().optional(),
});
export type CodexOptions = z.infer<typeof CodexOptions>;

/** A file Codex doesn't read itself, as text or input for it. */
export type CodexConvertFile = (
	file: LanguageModelV4FilePart,
) =>
	| string
	| CodexUserInput[]
	| undefined
	| Promise<string | CodexUserInput[] | undefined>;

export interface CodexProviderSettings {
	/**
	 * How to reach Codex: a host, or a function that makes one on first use.
	 * Defaults to running its app-server from this process (Node and Bun).
	 * Across a process or thread boundary, pass `connectCodexHost(channel)`.
	 */
	host?: CodexHost | (() => CodexHost | Promise<CodexHost>);
	/** Options for every call, under each model's and each call's own. */
	defaults?: CodexOptions;
	/**
	 * How long a thread waits, unused, for the next step before it is
	 * unloaded. It can still be resumed from disk after. Default 10 minutes.
	 */
	sessionTtlMs?: number;
	/** Files Codex doesn't take as they are (anything but images and text). */
	convertFile?: CodexConvertFile;
	/**
	 * The conversation before a thread, for one that can't be resumed, as
	 * text the model reads. Defaults to a plain transcript.
	 */
	transcript?: (messages: LanguageModelV4Prompt) => string | undefined;
}
