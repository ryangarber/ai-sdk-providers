import type { AfmAvailability, AfmizeClient } from "@afmize/node";
import {
	type LanguageModelV4,
	type LanguageModelV4Content,
	type LanguageModelV4StreamPart,
	NoSuchModelError,
	type ProviderV4,
	type SharedV4Warning,
} from "@ai-sdk/provider";
import {
	convertAsyncIteratorToReadableStream,
	createNullLanguageModelUsage,
	parseProviderOptions,
} from "@ai-sdk/provider-utils";
import { convertToAfmRequest } from "./convert.ts";
import { convertEvents } from "./stream.ts";
import {
	type AfmModelId,
	AfmOptions,
	type AfmProviderSettings,
} from "./types.ts";

const provider = "afm";

const models: readonly AfmModelId[] = ["on-device", "private-cloud-compute"];

export interface AfmProvider extends ProviderV4 {
	(modelId: AfmModelId): LanguageModelV4;
	languageModel(modelId: AfmModelId): LanguageModelV4;
	/** Which models this device can run, and why not. */
	availability(): Promise<AfmAvailability>;
}

/**
 * Apple Foundation Models, through afmize: `@afmize/node` by default, or the
 * client in `settings.afmize`.
 */
export function createAfmProvider(
	settings: AfmProviderSettings = {},
): AfmProvider {
	let client: Promise<AfmizeClient> | undefined;
	// `@afmize/node` resolves to a stub in a browser build, so the default
	// costs a webview nothing.
	const getClient = () =>
		(client ??= (async () => {
			const { afmize } = settings;
			if (typeof afmize === "function") return afmize();
			if (afmize) return afmize;
			return (await import("@afmize/node")).createAfmize();
		})());

	const languageModel = (modelId: AfmModelId): LanguageModelV4 => {
		if (!models.includes(modelId)) {
			throw new NoSuchModelError({ modelId, modelType: "languageModel" });
		}
		return {
			specificationVersion: "v4",
			provider,
			modelId,
			// afmize fetches images itself.
			supportedUrls: { "image/*": [/^https?:\/\//, /^file:/] },

			async doStream(options) {
				const { request, warnings } = convertToAfmRequest(
					modelId,
					options,
					await parseProviderOptions({
						provider,
						providerOptions: options.providerOptions,
						schema: AfmOptions,
					}),
				);
				const events = (await getClient()).stream(request, {
					signal: options.abortSignal,
				});
				return {
					stream: convertAsyncIteratorToReadableStream(
						convertEvents(events, warnings, options.includeRawChunks),
					),
					request: { body: request },
				};
			},

			async doGenerate(options) {
				const { stream, request } = await this.doStream(options);

				const content: LanguageModelV4Content[] = [];
				/** Text and reasoning blocks in `content`, filled in as they stream. */
				const blocks = new Map<string, { text: string }>();
				let warnings: SharedV4Warning[] = [];
				let finish:
					| Extract<LanguageModelV4StreamPart, { type: "finish" }>
					| undefined;
				let error: unknown;

				for await (const part of stream) {
					switch (part.type) {
						case "stream-start":
							warnings = part.warnings;
							break;
						case "text-start":
						case "reasoning-start": {
							const block = {
								type: part.type === "text-start" ? "text" : "reasoning",
								text: "",
							} as const satisfies LanguageModelV4Content;
							const filled = { ...block };
							blocks.set(part.id, filled);
							content.push(filled);
							break;
						}
						case "text-delta":
						case "reasoning-delta": {
							const block = blocks.get(part.id);
							if (block) block.text += part.delta;
							break;
						}
						case "file":
						case "tool-call":
							content.push(part);
							break;
						case "error":
							error ??= part.error;
							break;
						case "finish":
							finish = part;
							break;
					}
				}

				// Nothing to return, only why: as `generateText` would see it thrown.
				if (error && !content.length) throw error;

				return {
					content,
					finishReason: finish?.finishReason ?? {
						unified: "interrupted",
						raw: undefined,
					},
					usage: finish?.usage ?? createNullLanguageModelUsage(),
					warnings,
					request,
				};
			},
		};
	};

	const afm = (modelId: AfmModelId) => languageModel(modelId);

	return Object.assign(afm, {
		specificationVersion: "v4" as const,
		languageModel,
		availability: async () => (await getClient()).availability(),
		embeddingModel: (modelId: string) => {
			throw new NoSuchModelError({ modelId, modelType: "embeddingModel" });
		},
		imageModel: (modelId: string) => {
			throw new NoSuchModelError({ modelId, modelType: "imageModel" });
		},
	});
}

/** Apple Foundation Models through `@afmize/node`. */
export const afm: AfmProvider = createAfmProvider();
