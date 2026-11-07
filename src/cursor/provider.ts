import {
	type LanguageModelV4,
	NoSuchModelError,
	type ProviderV4,
} from "@ai-sdk/provider";
import type { CursorHost, CursorStatus } from "./host/protocol.ts";
import { createCursorModel, SessionRegistry } from "./model.ts";
import type {
	CursorModelId,
	CursorOptions,
	CursorProviderSettings,
} from "./options.ts";
import { cursorTools } from "./tools.ts";

export interface CursorProvider extends ProviderV4 {
	(modelId: CursorModelId, defaults?: CursorOptions): LanguageModelV4;
	languageModel(
		modelId: CursorModelId,
		defaults?: CursorOptions,
	): LanguageModelV4;
	/** Cursor's own tools, to declare in a call's `tools`. */
	tools: typeof cursorTools;
	/** Whether there is a working API key, and the models the account offers. */
	status(): Promise<CursorStatus>;
	/**
	 * Closes an open agent, or all of them. They can still be resumed from
	 * where they were saved.
	 */
	close(agentId?: string): Promise<void>;
}

/**
 * Cursor's agent, through the Cursor SDK's local agents, as a language model:
 * one step of its agent loop per `doStream`, continued by the next call from
 * where it left. Runs in this process by default (Node and Bun), or in any
 * host given.
 */
export function createCursor(
	settings: CursorProviderSettings = {},
): CursorProvider {
	let host: Promise<CursorHost> | undefined;
	const getHost = () =>
		(host ??= (async () => {
			const chosen = settings.host;
			if (typeof chosen === "function") return chosen();
			if (chosen) return chosen;
			// Imported only when used, so a bundle that brings its own host
			// never loads the Cursor SDK.
			const { createLocalHost } = await import("./host/local.ts");
			return createLocalHost({ apiKey: settings.apiKey });
		})());
	const registry = new SessionRegistry();

	const languageModel = (modelId: CursorModelId, defaults?: CursorOptions) => {
		if (!modelId)
			throw new NoSuchModelError({ modelId, modelType: "languageModel" });
		return createCursorModel({
			modelId,
			defaults,
			settings,
			getHost,
			registry,
		});
	};

	const provider = (modelId: CursorModelId, defaults?: CursorOptions) =>
		languageModel(modelId, defaults);

	return Object.assign(provider, {
		specificationVersion: "v4" as const,
		languageModel,
		tools: cursorTools,
		status: async () => (await getHost()).status(),
		close: async (agentId?: string) => {
			const sessions = agentId
				? [registry.get(agentId)].filter((s) => s !== undefined)
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

/** Cursor in this process, on `CURSOR_API_KEY` or a saved login. */
export const cursor: CursorProvider = createCursor();
