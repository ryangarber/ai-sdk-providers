import type { AfmEvent, AfmUsage } from "@afmize/node";
import type {
	LanguageModelV4FinishReason,
	LanguageModelV4StreamPart,
	LanguageModelV4Usage,
	SharedV4Warning,
} from "@ai-sdk/provider";

/** A stream's `error` event, as the error the AI SDK reports. */
export class AfmError extends Error {
	override name = "AfmError";
	constructor(
		/** afmize's error code, e.g. `guardrail-violation` or `context-size-exceeded`. */
		readonly code: string,
		message: string,
	) {
		super(message);
	}
}

const finishReasons: Record<
	string,
	LanguageModelV4FinishReason["unified"] | undefined
> = {
	stop: "stop",
	"tool-calls": "tool-calls",
	error: "error",
	other: "other",
};

/** Error codes that say more about why a stream stopped than `error`. */
const errorReasons: Record<
	string,
	LanguageModelV4FinishReason["unified"] | undefined
> = {
	"context-size-exceeded": "length",
	"guardrail-violation": "content-filter",
	refusal: "content-filter",
};

const convertUsage = (usage: AfmUsage | undefined): LanguageModelV4Usage => ({
	inputTokens: {
		total: usage?.inputTokens,
		noCache: usage && usage.inputTokens - usage.cachedInputTokens,
		cacheRead: usage?.cachedInputTokens,
		cacheWrite: undefined,
	},
	outputTokens: {
		total: usage?.outputTokens,
		text: usage && usage.outputTokens - usage.reasoningTokens,
		reasoning: usage?.reasoningTokens,
	},
	raw: usage && { ...usage },
});

/**
 * afmize's events as the AI SDK's stream parts: each text and reasoning
 * block gets its start and end, and each tool call its input.
 */
export async function* convertEvents(
	events: AsyncIterable<AfmEvent>,
	warnings: SharedV4Warning[],
	includeRawChunks = false,
): AsyncGenerator<LanguageModelV4StreamPart> {
	/** What each open block has sent so far, for a `*-replace` to amend. */
	const open = {
		text: new Map<string, string>(),
		reasoning: new Map<string, string>(),
	};
	let errorReason: LanguageModelV4FinishReason["unified"] | undefined;
	let started = false;

	function* end(
		kind: "text" | "reasoning",
	): Generator<LanguageModelV4StreamPart> {
		for (const id of open[kind].keys()) yield { type: `${kind}-end`, id };
		open[kind].clear();
	}

	function* delta(
		kind: "text" | "reasoning",
		id: string,
		delta: string,
	): Generator<LanguageModelV4StreamPart> {
		const sent = open[kind].get(id);
		if (sent === undefined) {
			// One block of each kind at a time; text follows its reasoning.
			yield* end(kind);
			if (kind === "text") yield* end("reasoning");
			yield { type: `${kind}-start`, id };
		}
		open[kind].set(id, (sent ?? "") + delta);
		if (delta) yield { type: `${kind}-delta`, id, delta };
	}

	function* replace(
		kind: "text" | "reasoning",
		id: string,
		text: string,
	): Generator<LanguageModelV4StreamPart> {
		const sent = open[kind].get(id) ?? "";
		if (text.startsWith(sent)) {
			yield* delta(kind, id, text.slice(sent.length));
			return;
		}
		// Sent text can't be taken back: the replacement is a block of its own.
		yield* end(kind);
		yield* delta(kind, `${id}:replaced`, text);
	}

	for await (const event of events) {
		if (includeRawChunks) yield { type: "raw", rawValue: event };

		switch (event.type) {
			case "stream-start":
				started = true;
				yield { type: "stream-start", warnings };
				break;
			case "text-delta":
			case "reasoning-delta":
				yield* delta(
					event.type === "text-delta" ? "text" : "reasoning",
					event.id,
					event.delta,
				);
				break;
			case "text-replace":
			case "reasoning-replace":
				yield* replace(
					event.type === "text-replace" ? "text" : "reasoning",
					event.id,
					event.text,
				);
				break;
			case "file":
				yield {
					type: "file",
					mediaType: event.mediaType,
					data: /^[a-z][a-z\d+.-]*:/i.test(event.data)
						? { type: "url", url: new URL(event.data) }
						: { type: "data", data: event.data },
				};
				break;
			case "tool-call":
				yield* end("reasoning");
				yield* end("text");
				yield {
					type: "tool-input-start",
					id: event.toolCallId,
					toolName: event.toolName,
				};
				yield {
					type: "tool-input-delta",
					id: event.toolCallId,
					delta: event.input,
				};
				yield { type: "tool-input-end", id: event.toolCallId };
				yield {
					type: "tool-call",
					toolCallId: event.toolCallId,
					toolName: event.toolName,
					input: event.input,
				};
				break;
			case "error":
				errorReason ??= errorReasons[event.code];
				yield { type: "error", error: new AfmError(event.code, event.message) };
				break;
			case "finish":
				if (!started) yield { type: "stream-start", warnings };
				yield* end("reasoning");
				yield* end("text");
				yield {
					type: "finish",
					finishReason: {
						unified:
							(event.finishReason === "error" ? errorReason : undefined) ??
							finishReasons[event.finishReason] ??
							"other",
						raw: event.finishReason,
					},
					usage: convertUsage(event.usage),
				};
				return;
		}
	}
}
