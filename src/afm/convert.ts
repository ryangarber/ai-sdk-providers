import type { AfmMessage, AfmPart, AfmRequest } from "@afmize/node";
import type {
	LanguageModelV4CallOptions,
	LanguageModelV4FilePart,
	LanguageModelV4FunctionTool,
	SharedV4Warning,
} from "@ai-sdk/provider";
import { convertToBase64 } from "@ai-sdk/provider-utils";
import type { AfmModelId, AfmOptions } from "./types.ts";

type Reasoning = NonNullable<LanguageModelV4CallOptions["reasoning"]>;

/** The call's `reasoning`, as afmize's levels; none for the model's default. */
const reasoningLevels: Record<Reasoning, AfmRequest["reasoningLevel"]> = {
	"provider-default": undefined,
	none: undefined,
	minimal: "light",
	low: "light",
	medium: "moderate",
	high: "deep",
	xhigh: "deep",
	max: "deep",
};

const unsupportedSettings = [
	"topP",
	"topK",
	"stopSequences",
	"presencePenalty",
	"frequencyPenalty",
	"seed",
] as const satisfies (keyof LanguageModelV4CallOptions)[];

/** A file part as afmize takes it, or a warning why it can't. */
function convertFile(
	part: Pick<LanguageModelV4FilePart, "data" | "mediaType" | "filename">,
): AfmPart | Extract<SharedV4Warning, { type: "unsupported" }> {
	const { data } = part;
	if (data.type === "text") return { type: "text", text: data.text };
	if (data.type === "reference") {
		return {
			type: "unsupported",
			feature: "file references",
			details: "afmize takes file data or URLs",
		};
	}
	// afmize decodes any image format, so a bare `image` is any of them.
	const mediaType =
		part.mediaType === "image" || part.mediaType === "image/*"
			? "image/*"
			: part.mediaType;
	if (!mediaType.startsWith("image/")) {
		return {
			type: "unsupported",
			feature: `${part.mediaType} files`,
			details: "Apple Foundation Models only take images",
		};
	}
	return {
		type: "file",
		mediaType,
		data:
			data.type === "url" ? data.url.toString() : convertToBase64(data.data),
		filename: part.filename,
	};
}

/** The AI SDK's call, as an afmize request, with what it had to leave out. */
export function convertToAfmRequest(
	model: AfmModelId,
	options: LanguageModelV4CallOptions,
	providerOptions: AfmOptions | undefined,
): { request: AfmRequest; warnings: SharedV4Warning[] } {
	const warnings: SharedV4Warning[] = [];

	for (const setting of unsupportedSettings) {
		if (options[setting] !== undefined)
			warnings.push({ type: "unsupported", feature: setting });
	}
	if (options.responseFormat?.type === "json") {
		warnings.push({
			type: "unsupported",
			feature: "responseFormat",
			details: "JSON output is not supported; the response is plain text",
		});
	}

	const messages: AfmMessage[] = options.prompt.map((message) => {
		const parts: AfmPart[] = [];
		if (message.role === "system") {
			parts.push({ type: "text", text: message.content });
			return { role: "system", parts };
		}
		for (const part of message.content) {
			switch (part.type) {
				case "text":
				case "reasoning":
					parts.push({ type: part.type, text: part.text });
					break;
				case "file": {
					const file = convertFile(part);
					if (file.type === "unsupported") warnings.push(file);
					else parts.push(file);
					break;
				}
				case "tool-call":
					parts.push({
						type: "tool-call",
						toolCallId: part.toolCallId,
						toolName: part.toolName,
						input: part.input,
					});
					break;
				case "tool-result":
					parts.push({
						type: "tool-result",
						toolCallId: part.toolCallId,
						toolName: part.toolName,
						output: part.output,
					});
					break;
				default:
					warnings.push({
						type: "unsupported",
						feature: `${part.type} parts`,
					});
			}
		}
		return { role: message.role, parts };
	});

	let tools = options.tools?.filter(
		(tool): tool is LanguageModelV4FunctionTool => {
			if (tool.type === "function") return true;
			warnings.push({
				type: "unsupported",
				feature: `provider tool ${tool.id}`,
			});
			return false;
		},
	);

	let toolChoice: AfmRequest["toolChoice"];
	const choice = options.toolChoice;
	if (choice?.type === "tool") {
		// afmize can require a call, not a tool; so only that tool is offered.
		tools = tools?.filter((tool) => tool.name === choice.toolName);
		toolChoice = "required";
	} else {
		toolChoice = choice?.type;
	}

	const request: AfmRequest = {
		model,
		messages,
		temperature: options.temperature,
		maximumResponseTokens: options.maxOutputTokens,
		reasoningLevel:
			providerOptions?.reasoningLevel ??
			(options.reasoning && reasoningLevels[options.reasoning]),
	};
	if (tools?.length) {
		request.tools = tools.map((tool) => ({
			name: tool.name,
			description: tool.description,
			inputSchema: tool.inputSchema,
		}));
	}
	// Without tools, there is no choice to make (and macOS 26 rejects one).
	if (request.tools && toolChoice) request.toolChoice = toolChoice;

	return { request, warnings };
}
