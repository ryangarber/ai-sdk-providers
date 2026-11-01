import type {
	LanguageModelV4Prompt,
	LanguageModelV4ToolResultOutput,
} from "@ai-sdk/provider";

const clip = (text: string, max = 4000) =>
	text.length > max
		? `${text.slice(0, max)}… (${text.length - max} more)`
		: text;

const stringify = (value: unknown) =>
	typeof value === "string" ? value : JSON.stringify(value);

/** A tool result's output, as text in a transcript. */
export const outputText = (output: LanguageModelV4ToolResultOutput) => {
	switch (output.type) {
		case "text":
		case "error-text":
			return output.value;
		case "json":
		case "error-json":
			return JSON.stringify(output.value);
		case "execution-denied":
			return `(denied${output.reason ? `: ${output.reason}` : ""})`;
		case "content":
			return output.value
				.map((part) =>
					part.type === "text"
						? part.text
						: part.type === "file"
							? `[file${part.filename ? `: ${part.filename}` : ""} (${part.mediaType})]`
							: "",
				)
				.join("\n");
	}
};

/**
 * The conversation as text the model reads, for a session that starts over:
 * one that moved devices, or whose earlier turns came from another model.
 */
export function defaultTranscript(
	messages: LanguageModelV4Prompt,
): string | undefined {
	const sections = messages.flatMap((message) => {
		if (message.role === "system") return [];
		const lines = (
			message.role === "tool" ? message.content : message.content
		).flatMap((part): string[] => {
			switch (part.type) {
				case "text":
					return part.text ? [part.text] : [];
				case "file":
					return [
						`[file${part.filename ? `: ${part.filename}` : ""} (${part.mediaType})]`,
					];
				case "tool-call":
					return [`[called ${part.toolName}: ${clip(stringify(part.input))}]`];
				case "tool-result":
					return [
						`[${part.toolName} returned: ${clip(outputText(part.output))}]`,
					];
				default:
					return [];
			}
		});
		if (!lines.length) return [];
		const author = message.role === "user" ? "User" : "Assistant";
		return [`## ${author}\n\n${lines.join("\n\n")}`];
	});
	if (!sections.length) return undefined;
	return [
		"<conversation_history>",
		"This conversation started before this session. What was said so far:",
		...sections,
		"</conversation_history>",
	].join("\n\n");
}
