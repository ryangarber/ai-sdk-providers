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
import type {
	CodexToolContent,
	CodexUserInput,
	CodexUserInputQuestion,
} from "./host/app-server.ts";
import type { CodexConvertFile } from "./options.ts";

export { defaultTranscript } from "#shared/transcript.ts";

/** The key of this provider's options and metadata. */
export const METADATA_KEY = "codex";

/** What this provider puts in the metadata of what it streams. */
export interface CodexPartMetadata {
	threadId: string;
	/** The Codex turn the part is from. */
	turnId?: string;
	/** The last Responses API call of the step it was streamed in. */
	cursor?: string;
	/** Whether the turn had ended with that step. */
	turnEnded?: boolean;
	[key: string]: unknown;
}

/** Where in a Codex thread the conversation was left. */
export interface Cursor {
	/** The index of the message it was found in. */
	index: number;
	threadId: string;
	turnId: string;
	/** The response the step ended after. */
	cursor: string;
	turnEnded: boolean;
	/** What the thread was started with. */
	fingerprint?: string;
	model?: string;
}

const metadataOf = (options: SharedV4ProviderOptions | undefined) => {
	const value = options?.[METADATA_KEY] as JSONObject | undefined;
	return typeof value?.threadId === "string" ? value : undefined;
};

/**
 * The last place the conversation was left in a thread: the last part of the
 * last assistant message this provider made that names one.
 */
export function findCursor(prompt: LanguageModelV4Prompt): Cursor | undefined {
	for (let index = prompt.length - 1; index >= 0; index--) {
		const message = prompt[index];
		if (message.role !== "assistant") continue;
		for (let i = message.content.length - 1; i >= 0; i--) {
			const metadata = metadataOf(message.content[i].providerOptions);
			if (
				metadata &&
				typeof metadata.cursor === "string" &&
				typeof metadata.turnId === "string"
			)
				return {
					index,
					threadId: metadata.threadId as string,
					turnId: metadata.turnId,
					cursor: metadata.cursor,
					turnEnded: metadata.turnEnded === true,
					...(typeof metadata.fingerprint === "string" && {
						fingerprint: metadata.fingerprint,
					}),
					...(typeof metadata.model === "string" && { model: metadata.model }),
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
	 * which the thread never saw.
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

/** A call made in the thread up to the cursor. */
export interface MadeCall {
	id: string;
	name: string;
	input: unknown;
	providerExecuted: boolean;
}

/**
 * Calls made in the turn up to the cursor that it never saw answered, in
 * order: they need a result before the thread can go on from there.
 */
export function getDangling(
	prompt: LanguageModelV4Prompt,
	cursor: Cursor,
): MadeCall[] {
	const calls = new Map<string, MadeCall>();
	for (const message of prompt.slice(0, cursor.index + 1)) {
		if (message.role === "user") calls.clear();
		if (message.role === "assistant")
			for (const part of message.content) {
				if (part.type === "tool-call" && metadataOf(part.providerOptions))
					calls.set(part.toolCallId, {
						id: part.toolCallId,
						name: part.toolName,
						input: part.input,
						providerExecuted: part.providerExecuted === true,
					});
				if (part.type === "tool-result") calls.delete(part.toolCallId);
			}
		if (message.role === "tool")
			for (const part of message.content)
				if (part.type === "tool-result") calls.delete(part.toolCallId);
	}
	return [...calls.values()];
}

/**
 * The call an approval is about: its id, or the call's id before a `#` when
 * a call is asked about more than once.
 */
export const approvalCall = (approvalId: string) => approvalId.split("#")[0];

const decode = (data: Uint8Array | string) =>
	typeof data === "string"
		? new TextDecoder().decode(
				Uint8Array.from(atob(data), (char) => char.charCodeAt(0)),
			)
		: new TextDecoder().decode(data);

const isImage = (mediaType: string) => mediaType.startsWith("image");

const text = (value: string): CodexUserInput => ({
	type: "text",
	text: value,
	text_elements: [],
});

const imageType = (mediaType: string) =>
	mediaType === "image" || mediaType === "image/*" ? "image/png" : mediaType;

/** A file as Codex takes it, or a warning why it can't. */
async function convertFile(
	part: LanguageModelV4FilePart,
	convert: CodexConvertFile | undefined,
): Promise<CodexUserInput[] | SharedV4Warning> {
	const { data, mediaType } = part;
	if (data.type === "text") return [text(data.text)];
	if (data.type !== "reference") {
		if (isImage(mediaType))
			return [
				{
					type: "image",
					url:
						data.type === "url"
							? data.url.toString()
							: `data:${imageType(mediaType)};base64,${convertToBase64(data.data)}`,
				},
			];
		if (data.type === "data" && mediaType.startsWith("text/"))
			return [text(decode(data.data))];
	}
	const converted = await convert?.(part);
	if (typeof converted === "string") return [text(converted)];
	if (converted) return converted;
	return {
		type: "unsupported",
		feature: `${mediaType} files`,
		details: "Codex takes images and text",
	};
}

/** What the user said, as Codex's input, with what had to be left out. */
export async function toInput(
	messages: LanguageModelV4Message[],
	options: {
		convertFile?: CodexConvertFile;
		warnings: SharedV4Warning[];
		transcript: (messages: LanguageModelV4Prompt) => string | undefined;
	},
): Promise<CodexUserInput[]> {
	const input: CodexUserInput[] = [];
	let others: LanguageModelV4Message[] = [];
	const flush = () => {
		const value = others.length ? options.transcript(others) : undefined;
		if (value) input.push(text(value));
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
				const last = input.at(-1);
				// Adjacent text reads as one, as it was written.
				if (last?.type === "text") last.text = `${last.text}\n${part.text}`;
				else if (part.text) input.push(text(part.text));
				continue;
			}
			const converted = await convertFile(part, options.convertFile);
			if (Array.isArray(converted)) input.push(...converted);
			else options.warnings.push(converted);
		}
	}
	flush();
	return input;
}

/** A tool result, as the dynamic tool response Codex reads. */
export function toToolContent(output: LanguageModelV4ToolResultOutput): {
	contentItems: CodexToolContent[];
	success: boolean;
} {
	const success = !(
		output.type === "error-text" ||
		output.type === "error-json" ||
		output.type === "execution-denied"
	);
	const contentItems: CodexToolContent[] =
		output.type === "content"
			? output.value.flatMap((part): CodexToolContent[] => {
					if (part.type === "text")
						return [{ type: "inputText", text: part.text }];
					if (part.type !== "file") return [];
					if (part.data.type === "text")
						return [{ type: "inputText", text: part.data.text }];
					if (part.data.type === "data" && isImage(part.mediaType))
						return [
							{
								type: "inputImage",
								imageUrl: `data:${imageType(part.mediaType)};base64,${convertToBase64(part.data.data)}`,
							},
						];
					if (part.data.type === "url" && isImage(part.mediaType))
						return [{ type: "inputImage", imageUrl: part.data.url.toString() }];
					if (part.data.type === "data" && part.mediaType.startsWith("text/"))
						return [{ type: "inputText", text: decode(part.data.data) }];
					return [
						{
							type: "inputText",
							text: `[file${part.filename ? `: ${part.filename}` : ""} (${part.mediaType})]`,
						},
					];
				})
			: [{ type: "inputText", text: outputText(output) }];
	return {
		contentItems: contentItems.length
			? contentItems
			: [{ type: "inputText", text: "(no output)" }],
		success,
	};
}

/** A tool result, as plain text for the model. */
export const toolText = (output: LanguageModelV4ToolResultOutput) =>
	toToolContent(output)
		.contentItems.map((item) =>
			item.type === "inputText" ? item.text : "[image]",
		)
		.join("\n");

const parseJson = (value: string) => {
	const body = value.trim().replace(/^```(?:json)?\s*([\s\S]*?)\s*```$/, "$1");
	try {
		return JSON.parse(body);
	} catch {
		return undefined;
	}
};

/**
 * The answers a `request_user_input` result holds, as Codex takes them: by
 * question id. The result is `{ answers }` keyed by question id or text, as
 * JSON or as text holding it (fenced or not).
 */
export function parseAnswers(
	output: LanguageModelV4ToolResultOutput,
	questions: CodexUserInputQuestion[],
): Record<string, { answers: string[] }> | undefined {
	const value =
		output.type === "json"
			? output.value
			: output.type === "text"
				? parseJson(output.value)
				: output.type === "content"
					? parseJson(
							output.value
								.map((part) => (part.type === "text" ? part.text : ""))
								.join(""),
						)
					: undefined;
	const answers = (value as { answers?: unknown } | undefined)?.answers;
	if (!answers || typeof answers !== "object") return undefined;
	const result: Record<string, { answers: string[] }> = {};
	for (const [key, answer] of Object.entries(answers)) {
		const question = questions.find(
			(question) => question.id === key || question.question === key,
		);
		const list = (Array.isArray(answer) ? answer : [answer])
			.filter((entry) => entry !== undefined && entry !== null)
			.map(String);
		result[question?.id ?? key] = { answers: list };
	}
	return result;
}
