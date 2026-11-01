/**
 * Drives the Codex provider against the installed, signed-in `codex` through
 * `streamText`, one step per call, as a caller with its own tool loop does:
 * `node scripts/codex-e2e.ts [cwd] [model]`. With `EXTRAS=1`, it tries
 * `apply_patch`, structured output, an abort and a subagent instead.
 */
import { type ModelMessage, streamText, tool } from "ai";
import { z } from "zod";
import { createCodex } from "#codex/provider.ts";

const cwd = process.argv[2] ?? "/tmp/codex-e2e";
const modelId = process.argv[3] ?? "gpt-6-luna";
const fs = await import("node:fs/promises");
await fs.rm(cwd, { recursive: true, force: true });
await fs.mkdir(cwd, { recursive: true });

const weather = tool({
	description: "The weather in a city",
	inputSchema: z.object({ city: z.string() }),
});
const time = tool({
	description: "The local time in a city",
	inputSchema: z.object({ city: z.string() }),
});

const baseTools = {
	get_weather: weather,
	shell: createCodex().tools.execCommand({}),
	patch: createCodex().tools.applyPatch({}),
	ask: createCodex().tools.requestUserInput({}),
};

/** Runs the conversation on until Codex is done with the turn. */
async function turn(
	codex: ReturnType<typeof createCodex>,
	messages: ModelMessage[],
	label: string,
	tools: Record<string, any> = baseTools,
) {
	for (let step = 0; step < 16; step++) {
		const result = streamText({
			model: codex(modelId),
			system: "You are terse.",
			messages,
			tools,
			providerOptions: {
				codex: {
					cwd,
					approvalPolicy: "untrusted",
					sandbox: "workspace-write",
					effort: "low",
				},
			},
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
			) {
				if (
					part.type === "tool-result" &&
					part.preliminary &&
					(part.output as any)?.type === "subagent"
				)
					seen.push(
						`subagent(${(part.output as any).status}:${(part.output as any).messages.length})`,
					);
				else
					seen.push(
						`${part.type}:${"toolName" in part ? part.toolName : (part as any).toolCall?.toolName}${"preliminary" in part && part.preliminary ? "(prelim)" : ""}`,
					);
			}
		}
		const finish = await result.finishReason;
		const metadata = (await result.providerMetadata)?.codex;
		console.log(
			`\n[${label} step ${step}] ${metadata?.route} ${finish} awaiting=${JSON.stringify(metadata?.awaiting)} ${seen.join(" ")}`,
		);
		messages.push(...(await result.response).messages);

		// The caller's part of the loop: its tools, approvals and answers.
		const content = (await result.content) ?? [];
		const replies: any[] = [];
		for (const part of content) {
			if (part.type === "tool-call" && !part.providerExecuted) {
				if (part.toolName === "get_weather" || part.toolName === "get_time")
					replies.push({
						type: "tool-result",
						toolCallId: part.toolCallId,
						toolName: part.toolName,
						output: {
							type: "text",
							value:
								part.toolName === "get_weather"
									? `Rainy in ${(part.input as any).city}`
									: `It is 9:41 in ${(part.input as any).city}`,
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
								answers: {
									[question.id]: question.options?.[0]?.label ?? "Tea",
								},
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

const codex = createCodex();
console.log(await codex.status());
const messages: ModelMessage[] = [
	{
		role: "user",
		content:
			"Call get_weather for Paris, and create notes.txt saying hi. Then reply in one sentence.",
	},
];
await turn(codex, messages, "A");
if (process.env.EXTRAS) {
	// apply_patch with its approval.
	messages.push({
		role: "user",
		content:
			"Use apply_patch (not the shell) to create todo.md containing one line: 'buy milk'. Then reply in one sentence.",
	});
	await turn(codex, messages, "P");
	// Structured output.
	const { Output } = await import("ai");
	const json = streamText({
		model: codex(modelId),
		messages: [{ role: "user", content: "Name a fruit and its color." }],
		output: Output.object({
			schema: z.object({ fruit: z.string(), color: z.string() }),
		}),
		providerOptions: { codex: { cwd, effort: "low" } },
	});
	console.log("[J]", JSON.stringify(await json.output));
	// Abort mid-step.
	const abort = new AbortController();
	const slow = streamText({
		model: codex(modelId),
		messages: [
			{ role: "user", content: "Write a 600-word story about a lighthouse." },
		],
		abortSignal: abort.signal,
		providerOptions: { codex: { cwd, effort: "low" } },
	});
	let chars = 0;
	try {
		for await (const part of slow.fullStream) {
			if (part.type === "text-delta") chars += part.text.length;
			if (chars > 80) abort.abort();
		}
	} catch {}
	console.log("[X] aborted after", chars, "chars");
	// A subagent.
	const agents: ModelMessage[] = [
		{
			role: "user",
			content:
				"Spawn one sub-agent with spawn_agent that runs `echo hello-from-child` in the shell and reports the output. Wait for it, then tell me what it said in one sentence.",
		},
	];
	await turn(codex, agents, "S", {
		...baseTools,
		agent: codex.tools.spawnAgent({}),
	});
	await codex.close();
	process.exit(0);
}

messages.push({
	role: "user",
	content:
		"Use request_user_input once to ask me whether I prefer tea or coffee, then run `cat notes.txt` in the shell and reply in one sentence.",
});
await turn(codex, messages, "B");

// Another tool: the thread is rebuilt with it.
messages.push({
	role: "user",
	content: "Call get_time for Paris and tell me in one sentence.",
});
await turn(codex, messages, "C", { ...baseTools, get_time: time });
await codex.close();

// A new provider has no live thread: the next turn rebuilds the saved one.
const later = createCodex();
messages.push({
	role: "user",
	content: "What did I say I prefer, and what was in notes.txt? One line.",
});
await turn(later, messages, "D", { ...baseTools, get_time: time });
await later.close();
process.exit(0);
