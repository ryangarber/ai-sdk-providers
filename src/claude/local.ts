import { createLocalHost } from "./host/local.ts";
import { type ClaudeCodeProvider, createClaudeCode } from "./provider.ts";

export * from "./host/local.ts";

/** Claude Code in this process, on the installed `claude`. */
export const claudeCode: ClaudeCodeProvider = createClaudeCode({
	host: createLocalHost(),
});
