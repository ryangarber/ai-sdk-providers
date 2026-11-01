import type {
	JSONObject,
	LanguageModelV4FilePart,
	LanguageModelV4Message,
	LanguageModelV4Prompt,
	LanguageModelV4ToolApprovalResponsePart,
	LanguageModelV4ToolResultOutput,
	LanguageModelV4ToolResultPart,
	SharedV4ProviderOptions,
	SharedV4Warning,
} from "@ai-sdk/provider";
import { convertToBase64 } from "@ai-sdk/provider-utils";
import { outputText } from "#shared/transcript.ts";
import type { ClaudeCodeToolContent } from "./host/protocol.ts";
import type {
	ClaudeCodeContentBlock,
	ClaudeCodeConvertFile,
} from "./options.ts";

export { defaultTranscript } from "#shared/transcript.ts";

/** The key of this provider's options and metadata. */
export const METADATA_KEY = "claudeCode";

/** What this provider puts in the metadata of what it streams. */
export interface ClaudeCodePartMetadata {
	sessionId: string;
	/** The session's last saved entry once the part was made, to resume at. */
	cursor?: string;
	[key: string]: unknown;
}

/** Where in a Claude Code session the conversation was left. */
export interface Cursor {
	/** The index of the message it was found in. */
	index: number;
	sessionId: string;
	cursor: string;
}

const metadataOf = (options: SharedV4ProviderOptions | undefined) => {
	const value = options?.[METADATA_KEY] as JSONObject | undefined;
	return typeof value?.sessionId === "string" ? value : undefined;
};

/**
 * The last place the conversation was left in a session: the last part of the
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
					sessionId: metadata.sessionId as string,
					cursor: metadata.cursor,
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
	approvals: LanguageModelV4ToolApprovalResponsePart[];
	/**
	 * What the user said, in order, with what other models said in between,
	 * which the session never saw.
	 */
	messages: LanguageModelV4Message[];
}

export function getTail(prompt: LanguageModelV4Prompt, from: number): Tail {
	const tail: Tail = { results: [], approvals: [], messages: [] };
	for (const message of prompt.slice(from)) {
		if (message.role === "system") continue;
		if (message.role === "tool") {
			for (const part of message.content)
				if (part.type === "tool-result") tail.results.push(part);
				else tail.approvals.push(part);
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

/**
 * Calls made in the session up to the cursor that it never saw answered, in
 * order: they need a result before the session can go on from there.
 */
export function getDangling(
	prompt: LanguageModelV4Prompt,
	cursor: Cursor,
): { id: string; name: string }[] {
	const calls = new Map<string, string>();
	for (const message of prompt.slice(0, cursor.index + 1)) {
		if (message.role === "user") calls.clear();
		if (message.role === "assistant")
			for (const part of message.content) {
				if (part.type === "tool-call" && metadataOf(part.providerOptions))
					calls.set(part.toolCallId, part.toolName);
				if (part.type === "tool-result") calls.delete(part.toolCallId);
			}
		if (message.role === "tool")
			for (const part of message.content)
				if (part.type === "tool-result") calls.delete(part.toolCallId);
	}
	return [...calls].map(([id, name]) => ({ id, name }));
}

const decode = (data: Uint8Array | string) =>
	typeof data === "string"
		? new TextDecoder().decode(
				Uint8Array.from(atob(data), (char) => char.charCodeAt(0)),
			)
		: new TextDecoder().decode(data);

const isImage = (mediaType: string) => mediaType.startsWith("image");

/** A file as Claude takes it, or a warning why it can't. */
async function convertFile(
	part: LanguageModelV4FilePart,
	convert: ClaudeCodeConvertFile | undefined,
): Promise<ClaudeCodeContentBlock[] | SharedV4Warning> {
	const { data, mediaType } = part;
	if (data.type === "text") return [{ type: "text", text: data.text }];
	if (data.type !== "reference") {
		if (isImage(mediaType) || mediaType === "application/pdf") {
			const source =
				data.type === "url"
					? { type: "url" as const, url: data.url.toString() }
					: {
							type: "base64" as const,
							media_type:
								mediaType === "image" || mediaType === "image/*"
									? "image/png"
									: mediaType,
							data: convertToBase64(data.data),
						};
			if (isImage(mediaType)) return [{ type: "image", source }];
			return [
				{
					type: "document",
					source:
						source.type === "url"
							? source
							: { ...source, media_type: "application/pdf" },
					...(part.filename ? { title: part.filename } : {}),
				},
			];
		}
		if (data.type === "data" && mediaType.startsWith("text/"))
			return [{ type: "text", text: decode(data.data) }];
	}
	const converted = await convert?.(part);
	if (typeof converted === "string") return [{ type: "text", text: converted }];
	if (converted) return converted;
	return {
		type: "unsupported",
		feature: `${mediaType} files`,
		details: "Claude Code takes images, PDFs and text",
	};
}

/** What the user said, as content blocks, with what had to be left out. */
export async function toBlocks(
	messages: LanguageModelV4Message[],
	options: {
		convertFile?: ClaudeCodeConvertFile;
		warnings: SharedV4Warning[];
		transcript: (messages: LanguageModelV4Prompt) => string | undefined;
	},
): Promise<ClaudeCodeContentBlock[]> {
	const blocks: ClaudeCodeContentBlock[] = [];
	let others: LanguageModelV4Message[] = [];
	const flush = () => {
		const text = others.length ? options.transcript(others) : undefined;
		if (text) blocks.push({ type: "text", text });
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
				const last = blocks.at(-1);
				// Adjacent text reads as one block, as it was written.
				if (last?.type === "text") last.text = `${last.text}\n${part.text}`;
				else if (part.text) blocks.push({ type: "text", text: part.text });
				continue;
			}
			const converted = await convertFile(part, options.convertFile);
			if (Array.isArray(converted)) blocks.push(...converted);
			else options.warnings.push(converted);
		}
	}
	flush();
	return blocks;
}

/**
 * The answers an AskUserQuestion's result holds, `{ answers, annotations? }`,
 * as JSON or as text holding it (fenced or not).
 */
export function parseAnswers(output: LanguageModelV4ToolResultOutput):
	| {
			answers: Record<string, string>;
			annotations?: Record<string, unknown>;
	  }
	| undefined {
	const fromText = (text: string) => {
		const body = text.trim().replace(/^```(?:json)?\s*([\s\S]*?)\s*```$/, "$1");
		try {
			return JSON.parse(body);
		} catch {
			return undefined;
		}
	};
	const value =
		output.type === "json"
			? output.value
			: output.type === "text"
				? fromText(output.value)
				: output.type === "content"
					? fromText(
							output.value
								.map((part) => (part.type === "text" ? part.text : ""))
								.join(""),
						)
					: undefined;
	const answers = (value as { answers?: unknown } | undefined)?.answers;
	if (!answers || typeof answers !== "object") return undefined;
	const annotations = (value as { annotations?: unknown }).annotations;
	return {
		answers: answers as Record<string, string>,
		...(typeof annotations === "object" &&
			annotations !== null && {
				annotations: annotations as Record<string, unknown>,
			}),
	};
}

/** A tool result, as the MCP result Claude Code reads. */
export function toToolContent(output: LanguageModelV4ToolResultOutput): {
	content: ClaudeCodeToolContent[];
	isError: boolean;
} {
	const isError =
		output.type === "error-text" ||
		output.type === "error-json" ||
		output.type === "execution-denied";
	const content: ClaudeCodeToolContent[] =
		output.type === "content"
			? output.value.flatMap((part): ClaudeCodeToolContent[] => {
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
		isError,
	};
}
