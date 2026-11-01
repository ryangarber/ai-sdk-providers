/**
 * Drives the Claude Code provider against the installed, signed-in `claude`
 * through `streamText`, one step per call, as a caller with its own tool loop
 * does: `node scripts/claude-e2e.ts`. Uses the account's Haiku.
 */
import { type ModelMessage, streamText, tool } from "ai";
import { z } from "zod";
import { createClaudeCode } from "#claude/provider.ts";

const cwd = process.argv[2] ?? "/tmp/cc-e2e";
await import("node:fs/promises").then((fs) =>
	fs.mkdir(cwd, { recursive: true }),
);

const tools = {
	get_weather: tool({
		description: "The weather in a city",
		inputSchema: z.object({ city: z.string() }),
	}),
	Bash: createClaudeCode().tools.bash({}),
	Write: createClaudeCode().tools.write({}),
	ask: createClaudeCode().tools.askUserQuestion({}),
};

/** Runs the conversation on until Claude Code is done with the turn. */
async function turn(
	claudeCode: ReturnType<typeof createClaudeCode>,
	messages: ModelMessage[],
	label: string,
) {
	for (let step = 0; step < 12; step++) {
		const result = streamText({
			model: claudeCode("haiku"),
			system: "You are terse.",
			messages,
			tools,
			providerOptions: { claudeCode: { cwd, permissionMode: "default" } },
		});
		const seen: string[] = [];
		for await (const part of result.fullStream) {
			if (part.type === "text-delta") process.stdout.write(part.text);
			else if (part.type === "error") console.error("\n[error]", part.error);
			else if (
				[
					"tool-call",
					"tool-result",
					"tool-approval-request",
					"tool-error",
				].includes(part.type)
			)
				seen.push(
					`${part.type}:${"toolName" in part ? part.toolName : (part as any).toolCall?.toolName}${"preliminary" in part && part.preliminary ? "(prelim)" : ""}`,
				);
		}
		const finish = await result.finishReason;
		const metadata = (await result.providerMetadata)?.claudeCode;
		console.log(
			`\n[${label} step ${step}] ${metadata?.route} ${finish} awaiting=${JSON.stringify(metadata?.awaiting)} ${seen.join(" ")}`,
		);
		messages.push(...(await result.response).messages);

		// The caller's part of the loop: its tools, approvals and answers.
		const content = (await result.content) ?? [];
		const replies: any[] = [];
		for (const part of content) {
			if (part.type === "tool-call" && !part.providerExecuted) {
				if (part.toolName === "get_weather")
					replies.push({
						type: "tool-result",
						toolCallId: part.toolCallId,
						toolName: part.toolName,
						output: {
							type: "text",
							value: `Rainy in ${(part.input as any).city}`,
						},
					});
				if (part.toolName === "ask") {
					const question = (part.input as any).questions?.[0];
					replies.push({
						type: "tool-result",
						toolCallId: part.toolCallId,
						toolName: part.toolName,
						output: {
							type: "json",
							value: {
								answers: { [question.question]: question.options[0].label },
							},
						},
					});
				}
			}
			if (part.type === "tool-approval-request")
				replies.push({
					type: "tool-approval-response",
					approvalId: part.approvalId,
					approved: true,
					providerExecuted: true,
				});
		}
		if (replies.length) messages.push({ role: "tool", content: replies });
		if (finish !== "tool-calls") return;
	}
}

const claudeCode = createClaudeCode();
const messages: ModelMessage[] = [
	{
		role: "user",
		content:
			"Call get_weather for Paris, and use Write to create notes.txt saying hi. Then reply in one sentence.",
	},
];
await turn(claudeCode, messages, "A");

messages.push({
	role: "user",
	content:
		"Use the ask tool (AskUserQuestion) once to ask me whether I prefer tea or coffee, then run `cat notes.txt` with Bash and reply in one sentence.",
});
await turn(claudeCode, messages, "B");
await claudeCode.close();

// A new provider has no live session: the next turn forks the saved one.
const later = createClaudeCode();
messages.push({ role: "user", content: "What did I say I prefer? One word." });
await turn(later, messages, "C");
await later.close();
