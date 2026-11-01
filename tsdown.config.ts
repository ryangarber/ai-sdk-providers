import { defineConfig } from "tsdown";

export default defineConfig({
	entry: [
		"src/afm/index.ts",
		"src/claude/index.ts",
		"src/claude/local.ts",
		"src/codex/index.ts",
		"src/codex/local.ts",
	],
	format: "esm",
	platform: "node",
	dts: {
		enabled: true,
		sourcemap: true,
	},
	sourcemap: true,
	deps: { neverBundle: true },
});
