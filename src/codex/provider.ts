import {
	type LanguageModelV4,
	NoSuchModelError,
	type ProviderV4,
} from "@ai-sdk/provider";
import type {
	CodexHost,
	CodexStatus,
	CodexStatusOptions,
} from "./host/protocol.ts";
import { createCodexModel } from "./model.ts";
import type {
	CodexModelId,
	CodexOptions,
	CodexProviderSettings,
} from "./options.ts";
import { SessionRegistry } from "./session.ts";
import { codexTools } from "./tools.ts";

export interface CodexProvider extends ProviderV4 {
	(modelId: CodexModelId, defaults?: CodexOptions): LanguageModelV4;
	languageModel(
		modelId: CodexModelId,
		defaults?: CodexOptions,
	): LanguageModelV4;
	/** Codex's own tools, to declare in a call's `tools`. */
	tools: typeof codexTools;
	/**
	 * Whether `codex` is installed and signed in, and the models it offers
	 * the account.
	 */
	status(options?: CodexStatusOptions): Promise<CodexStatus>;
	/** Stops one subagent of a live thread, by the call that spawned it. */
	stopTask(threadId: string, toolCallId: string): Promise<void>;
	/**
	 * Unloads a live thread, or all of them. They can still be continued
	 * from where they were saved.
	 */
	close(threadId?: string): Promise<void>;
}

/**
 * Codex, through its app-server, as a language model: one Responses API call
 * of its agent loop per `doStream`, continued by the next call from where it
 * left. Runs `codex` from this process by default (Node and Bun), or in any
 * host given.
 */
export function createCodex(
	settings: CodexProviderSettings = {},
): CodexProvider {
	let host: Promise<CodexHost> | undefined;
	const getHost = () =>
		(host ??= (async () => {
			const chosen = settings.host;
			if (typeof chosen === "function") return chosen();
			if (chosen) return chosen;
			// Imported only when used, so a bundle that brings its own host
			// never loads Node's modules.
			const { createLocalHost } = await import("./host/local.ts");
			return createLocalHost();
		})());
	const registry = new SessionRegistry();

	const languageModel = (modelId: CodexModelId, defaults?: CodexOptions) => {
		if (!modelId)
			throw new NoSuchModelError({ modelId, modelType: "languageModel" });
		return createCodexModel({
			modelId,
			defaults,
			settings,
			getHost,
			registry,
		});
	};

	const provider = (modelId: CodexModelId, defaults?: CodexOptions) =>
		languageModel(modelId, defaults);

	return Object.assign(provider, {
		specificationVersion: "v4" as const,
		languageModel,
		tools: codexTools,
		status: async (options?: CodexStatusOptions) =>
			(await getHost()).status(options),
		stopTask: async (threadId: string, toolCallId: string) =>
			registry.get(threadId)?.stopTask(toolCallId),
		close: async (threadId?: string) => {
			const sessions = threadId
				? [registry.get(threadId)].filter((s) => s !== undefined)
				: registry.all();
			await Promise.all(sessions.map((session) => session.close()));
		},
		embeddingModel: (modelId: string) => {
			throw new NoSuchModelError({ modelId, modelType: "embeddingModel" });
		},
		imageModel: (modelId: string) => {
			throw new NoSuchModelError({ modelId, modelType: "imageModel" });
		},
	});
}

/** Codex from this process, on the installed `codex`. */
export const codex: CodexProvider = createCodex();
