import { createLocalHost } from "./host/local.ts";
import { type CodexProvider, createCodex } from "./provider.ts";

export * from "./host/local.ts";

/** Codex from this process, on the installed `codex`. */
export const codex: CodexProvider = createCodex({
	host: createLocalHost(),
});
