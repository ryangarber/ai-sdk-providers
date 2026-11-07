import type {
	JSONObject,
	LanguageModelV4FilePart,
	LanguageModelV4Message,
	LanguageModelV4Prompt,
	LanguageModelV4ToolResultOutput,
	LanguageModelV4ToolResultPart,
	SharedV4ProviderOptions,
	SharedV4Warning,
} from "@ai-sdk/provider";
import { convertToBase64 } from "@ai-sdk/provider-utils";
import { outputText } from "#shared/transcript.ts";
import type {
	CursorToolAnswer,
	CursorToolContent,
	CursorUserImage,
	CursorUserMessage,
} from "./host/protocol.ts";
import type { CursorConvertFile } from "./options.ts";

export { defaultTranscript } from "#shared/transcript.ts";

/** The key of this provider's options and metadata. */
export const METADATA_KEY = "cursor";

/** What this provider puts in the metadata of what it streams. */
export interface CursorPartMetadata {
	agentId: string;
	/** The run and step the part was made in: `<runId>#<step>`. */
	cursor?: string;
	[key: string]: unknown;
}

/** Where in a Cursor agent the conversation was left. */
export interface Cursor {
	/** The index of the message it was found in. */
	index: number;
	agentId: string;
	cursor: string;
	runId: string;
}

const metadataOf = (options: SharedV4ProviderOptions | undefined) => {
	const value = options?.[METADATA_KEY] as JSONObject | undefined;
	return typeof value?.agentId === "string" ? value : undefined;
};

/** The run a cursor is in. */
const runOf = (cursor: string) => cursor.split("#")[0];

/**
 * The last place the conversation was left in an agent: the last part of the
 * last assistant message this provider made that names one.
 */
export function findCursor(prompt: LanguageModelV4Prompt): Cursor | undefined {
	for (let index = prompt.length - 1; index >= 0; index--) {
		const message = prompt[index];
		if (message.role !== "assistant") continue;
		for (let i = message.content.length - 1; i >= 0; i--) {
			const metadata = metadataOf(message.content[i].providerOptions);
			if (metadata && typeof metadata.cursor === "string")
				return {
					index,
					agentId: metadata.agentId as string,
					cursor: metadata.cursor,
					runId: runOf(metadata.cursor),
				};
		}
	}
	return undefined;
}

/** Whether an assistant message came from this provider. */
const isOwn = (message: LanguageModelV4Message) =>
	message.role === "assistant" &&
	message.content.some((part) => metadataOf(part.providerOptions));

/** What came after the cursor, by what each has to go to. */
export interface Tail {
	results: LanguageModelV4ToolResultPart[];
	/**
	 * What the user said, in order, with what other models said in between,
	 * which the agent never saw.
	 */
	messages: LanguageModelV4Message[];
}

export function getTail(prompt: LanguageModelV4Prompt, from: number): Tail {
	const tail: Tail = { results: [], messages: [] };
	for (const message of prompt.slice(from)) {
		if (message.role === "system") continue;
		if (message.role === "tool") {
			for (const part of message.content)
				if (part.type === "tool-result") tail.results.push(part);
			continue;
		}
		if (message.role === "assistant" && isOwn(message)) continue;
		tail.messages.push(message);
	}
	return tail;
}

/** The call's system messages, as one prompt. */
export const getSystem = (prompt: LanguageModelV4Prompt) =>
	prompt
		.flatMap((message) => (message.role === "system" ? [message.content] : []))
		.join("\n\n") || undefined;

/** System instructions, as the top of a message to an agent with its own. */
export const withSystem = (system: string, text: string) =>
	`<system_instructions>\n${system}\n</system_instructions>${text ? `\n\n${text}` : ""}`;

const decode = (data: Uint8Array | string) =>
	typeof data === "string"
		? new TextDecoder().decode(
				Uint8Array.from(atob(data), (char) => char.charCodeAt(0)),
			)
		: new TextDecoder().decode(data);

const isImage = (mediaType: string) => mediaType.startsWith("image");

/** A file as Cursor takes it: text, images, or a warning why it can't. */
async function convertFile(
	part: LanguageModelV4FilePart,
	convert: CursorConvertFile | undefined,
): Promise<string | CursorUserImage[] | SharedV4Warning> {
	const { data, mediaType } = part;
	if (data.type === "text") return data.text;
	if (data.type === "url" && isImage(mediaType))
		return [{ url: data.url.toString() }];
	if (data.type === "data") {
		if (isImage(mediaType))
			return [
				{
					data: convertToBase64(data.data),
					mimeType:
						mediaType === "image" || mediaType === "image/*"
							? "image/png"
							: mediaType,
				},
			];
		if (mediaType.startsWith("text/")) return decode(data.data);
	}
	const converted = await convert?.(part);
	if (converted !== undefined) return converted;
	return {
		type: "unsupported",
		feature: `${mediaType} files`,
		details: "Cursor takes images and text",
	};
}

/** What the user said, as one message, with what had to be left out. */
export async function toMessage(
	messages: LanguageModelV4Message[],
	options: {
		convertFile?: CursorConvertFile;
		warnings: SharedV4Warning[];
		transcript: (messages: LanguageModelV4Prompt) => string | undefined;
	},
): Promise<CursorUserMessage> {
	const texts: string[] = [];
	const images: CursorUserImage[] = [];
	let others: LanguageModelV4Message[] = [];
	const flush = () => {
		const text = others.length ? options.transcript(others) : undefined;
		if (text) texts.push(text);
		others = [];
	};
	for (const message of messages) {
		if (message.role !== "user") {
			others.push(message);
			continue;
		}
		flush();
		for (const part of message.content) {
			if (part.type === "text") {
				if (part.text) texts.push(part.text);
				continue;
			}
			const converted = await convertFile(part, options.convertFile);
			if (typeof converted === "string") texts.push(converted);
			else if (Array.isArray(converted)) images.push(...converted);
			else options.warnings.push(converted);
		}
	}
	flush();
	return { text: texts.join("\n\n"), ...(images.length && { images }) };
}

/** A tool result, as the MCP result Cursor reads. */
export function toToolAnswer(
	output: LanguageModelV4ToolResultOutput,
): CursorToolAnswer {
	const isError =
		output.type === "error-text" ||
		output.type === "error-json" ||
		output.type === "execution-denied";
	const content: CursorToolContent[] =
		output.type === "content"
			? output.value.flatMap((part): CursorToolContent[] => {
					if (part.type === "text") return [{ type: "text", text: part.text }];
					if (part.type !== "file") return [];
					if (part.data.type === "text")
						return [{ type: "text", text: part.data.text }];
					if (part.data.type === "data" && isImage(part.mediaType))
						return [
							{
								type: "image",
								data: convertToBase64(part.data.data),
								mimeType: part.mediaType,
							},
						];
					if (part.data.type === "data" && part.mediaType.startsWith("text/"))
						return [{ type: "text", text: decode(part.data.data) }];
					return [
						{
							type: "text",
							text: `[file${part.filename ? `: ${part.filename}` : ""} (${part.mediaType})]`,
						},
					];
				})
			: [{ type: "text", text: outputText(output) }];
	return {
		content: content.length ? content : [{ type: "text", text: "(no output)" }],
		...(isError && { isError }),
	};
}
