import {
	type LanguageModelV4,
	NoSuchModelError,
	type ProviderV4,
} from "@ai-sdk/provider";
import type {
	ClaudeCodeHost,
	ClaudeCodeStatus,
	ClaudeCodeStatusOptions,
} from "./host/protocol.ts";
import { createClaudeCodeModel, SessionRegistry } from "./model.ts";
import type {
	ClaudeCodeModelId,
	ClaudeCodeOptions,
	ClaudeCodeProviderSettings,
} from "./options.ts";
import { claudeCodeTools } from "./tools.ts";

export interface ClaudeCodeProvider extends ProviderV4 {
	(modelId: ClaudeCodeModelId, defaults?: ClaudeCodeOptions): LanguageModelV4;
	languageModel(
		modelId: ClaudeCodeModelId,
		defaults?: ClaudeCodeOptions,
	): LanguageModelV4;
	/** Claude Code's own tools, to declare in a call's `tools`. */
	tools: typeof claudeCodeTools;
	/**
	 * Whether `claude` is installed and signed in, and the models, agents and
	 * output styles it offers the account.
	 */
	status(options?: ClaudeCodeStatusOptions): Promise<ClaudeCodeStatus>;
	/** Stops one subagent of a live session, by the call that started it. */
	stopTask(sessionId: string, toolCallId: string): Promise<void>;
	/**
	 * Closes a live session, or all of them. They can still be resumed from
	 * where they were saved.
	 */
	close(sessionId?: string): Promise<void>;
}

/**
 * Claude Code, through the Agent SDK, as a language model: one step of its
 * agent loop per `doStream`, continued by the next call from where it left.
 * Runs in this process by default (Node and Bun), or in any host given.
 */
export function createClaudeCode(
	settings: ClaudeCodeProviderSettings = {},
): ClaudeCodeProvider {
	let host: Promise<ClaudeCodeHost> | undefined;
	const getHost = () =>
		(host ??= (async () => {
			const chosen = settings.host;
			if (typeof chosen === "function") return chosen();
			if (chosen) return chosen;
			// Imported only when used, so a bundle that brings its own host
			// never loads the Agent SDK.
			const { createLocalHost } = await import("./host/local.ts");
			return createLocalHost();
		})());
	const registry = new SessionRegistry();

	const languageModel = (
		modelId: ClaudeCodeModelId,
		defaults?: ClaudeCodeOptions,
	) => {
		if (!modelId)
			throw new NoSuchModelError({ modelId, modelType: "languageModel" });
		return createClaudeCodeModel({
			modelId,
			defaults,
			settings,
			getHost,
			registry,
		});
	};

	const provider = (modelId: ClaudeCodeModelId, defaults?: ClaudeCodeOptions) =>
		languageModel(modelId, defaults);

	return Object.assign(provider, {
		specificationVersion: "v4" as const,
		languageModel,
		tools: claudeCodeTools,
		status: async (options?: ClaudeCodeStatusOptions) =>
			(await getHost()).status(options),
		stopTask: async (sessionId: string, toolCallId: string) =>
			registry.get(sessionId)?.stopTask(toolCallId),
		close: async (sessionId?: string) => {
			const sessions = sessionId
				? [registry.get(sessionId)].filter((s) => s !== undefined)
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

/** Claude Code in this process, on the installed `claude`. */
export const claudeCode: ClaudeCodeProvider = createClaudeCode();
