/**
 * Drives the Cursor provider against the real Cursor SDK through
 * `streamText`, one step per call, as a caller with its own tool loop does:
 * `node scripts/cursor-e2e.ts [cwd]`. Uses `CURSOR_API_KEY`, or the key
 * `Cursor.auth.login()` saved; `MODEL` picks the model (`composer-2.5`).
 */
import { mkdir, writeFile } from "node:fs/promises";
import { type ModelMessage, streamText, tool } from "ai";
import { z } from "zod";
import { createCursor } from "#cursor/provider.ts";

const cwd = process.argv[2] ?? "/tmp/cursor-e2e";
const modelId = process.env.MODEL ?? "composer-2.5";
await mkdir(cwd, { recursive: true });
await writeFile(`${cwd}/notes.txt`, "The secret word is marmalade.\n");

const tools = {
	get_weather: tool({
		description: "The weather in a city",
		inputSchema: z.object({ city: z.string() }),
	}),
	read: createCursor().tools.read({}),
};

/** Runs the conversation on until Cursor is done with the turn. */
async function turn(
	cursor: ReturnType<typeof createCursor>,
	messages: ModelMessage[],
	label: string,
	options: { abortAfterMs?: number } = {},
) {
	for (let step = 0; step < 12; step++) {
		const abort = new AbortController();
		if (options.abortAfterMs)
			setTimeout(() => abort.abort(), options.abortAfterMs);
		const result = streamText({
			model: cursor(modelId),
			system: "You are terse. Use the tools you are given.",
			messages,
			tools,
			...(process.env.EFFORT && { reasoning: process.env.EFFORT as "low" }),
			abortSignal: abort.signal,
			providerOptions: { cursor: { cwd } },
		});
		const seen: string[] = [];
		for await (const part of result.fullStream) {
			if (part.type === "text-delta") process.stdout.write(part.text);
			else if (part.type === "error") console.error("\n[error]", part.error);
			else if (part.type === "abort") seen.push("abort");
			else if (["tool-call", "tool-result", "tool-error"].includes(part.type))
				seen.push(
					`${part.type}:${"toolName" in part ? part.toolName : ""}${"preliminary" in part && part.preliminary ? "(prelim)" : ""}`,
				);
		}
		if (abort.signal.aborted) {
			console.log(`\n[${label} step ${step}] aborted ${seen.join(" ")}`);
			return;
		}
		const finish = await result.finishReason;
		const metadata = (await result.providerMetadata)?.cursor;
		const usage = await result.usage;
		console.log(
			`\n[${label} step ${step}] ${metadata?.route} ${finish} awaiting=${JSON.stringify(metadata?.awaiting)} in=${usage.inputTokens} out=${usage.outputTokens} ${seen.join(" ")}`,
		);
		messages.push(...(await result.response).messages);

		// The caller's part of the loop: its tools.
		const content = (await result.content) ?? [];
		const replies: any[] = [];
		for (const part of content)
			if (
				part.type === "tool-call" &&
				!part.providerExecuted &&
				part.toolName === "get_weather"
			)
				replies.push({
					type: "tool-result",
					toolCallId: part.toolCallId,
					toolName: part.toolName,
					output: {
						type: "text",
						value: `Rainy and 9C in ${(part.input as any).city}`,
					},
				});
		if (replies.length) messages.push({ role: "tool", content: replies });
		if (finish !== "tool-calls") return;
	}
}

const cursor = createCursor();
const status = await cursor.status();
console.log(
	`[status] loggedIn=${status.loggedIn} models=${status.models?.length}${status.error ? ` error=${status.error}` : ""}`,
);

const messages: ModelMessage[] = [
	{
		role: "user",
		content:
			"Call get_weather for Paris and Tokyo, and read notes.txt. Then reply in one sentence with the weather and the secret word.",
	},
];
await turn(cursor, messages, "A");

messages.push({
	role: "user",
	content: "Which city was it rainy in first? One word.",
});
await turn(cursor, messages, "B");
await cursor.close();

// A new provider has no open agent: the next turn resumes the saved one.
const later = createCursor();
messages.push({ role: "user", content: "What was the secret word? One word." });
await turn(later, messages, "C");

// Aborted mid-reply: the run is cancelled, and what it said is dropped.
const before = messages.length;
messages.push({
	role: "user",
	content: "Write a 1000-word story about the weather, without using tools.",
});
await turn(later, messages, "D", { abortAfterMs: 2500 });
messages.length = before;

// The agent went further than the conversation did: it starts over.
messages.push({ role: "user", content: "And the second city? One word." });
await turn(later, messages, "E");
await later.close();
process.exit(0);
