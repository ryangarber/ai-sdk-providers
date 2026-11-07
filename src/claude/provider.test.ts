import type {
	LanguageModelV4CallOptions,
	LanguageModelV4Message,
	LanguageModelV4Prompt,
	LanguageModelV4StreamPart,
} from "@ai-sdk/provider";
import fixture from "#claude/fixtures/mcp-and-bash.json" with { type: "json" };
import type {
	ClaudeCodeHost,
	ClaudeCodeHostAnswer,
	ClaudeCodeHostEvent,
	ClaudeCodeRunRequest,
} from "#claude/host/protocol.ts";
import { createClaudeCode } from "#claude/provider.ts";

interface FakeRun {
	id: string;
	request: ClaudeCodeRunRequest;
	emit: (...events: ClaudeCodeHostEvent[]) => void;
	sent: unknown[];
	answers: Map<string, ClaudeCodeHostAnswer>;
	interrupted: number;
	/** Resolves once question `id` is answered. */
	answered: (id: string) => Promise<ClaudeCodeHostAnswer>;
}

/** A host that records what it is given and emits what a test says. */
const fakeHost = (saved: string[] = []) => {
	const runs: FakeRun[] = [];
	const waiters = new Map<string, (answer: ClaudeCodeHostAnswer) => void>();
	const host: ClaudeCodeHost = {
		status: async () => ({ installed: true, loggedIn: true }),
		async start(request, onEvent) {
			const run: FakeRun = {
				id: `run-${runs.length}`,
				request,
				emit: (...events) => {
					for (const event of events) onEvent(event);
				},
				sent: [],
				answers: new Map(),
				interrupted: 0,
				answered: (id) =>
					run.answers.has(id)
						? Promise.resolve(run.answers.get(id) as ClaudeCodeHostAnswer)
						: new Promise((resolve) => waiters.set(`${run.id}:${id}`, resolve)),
			};
			runs.push(run);
			return run.id;
		},
		send: async (id, message) => {
			runs.find((run) => run.id === id)?.sent.push(message);
		},
		answer: async (id, question, answer) => {
			runs.find((run) => run.id === id)?.answers.set(question, answer);
			waiters.get(`${id}:${question}`)?.(answer);
		},
		update: async () => {},
		interrupt: async (id) => {
			const run = runs.find((run) => run.id === id);
			if (run) run.interrupted++;
		},
		stopTask: async () => {},
		close: async () => {},
		hasSession: async (session) => saved.includes(session),
	};
	return { host, runs };
};

/**
 * Plays a recording into a run, holding back what came after each question
 * until the provider answers it, as a real session would.
 */
const replay = async (run: FakeRun, events: ClaudeCodeHostEvent[]) => {
	const asked = new Map<string, string>();
	for (const event of events) {
		if (event.type === "tool-call" || event.type === "permission")
			if (event.toolUseId) asked.set(event.toolUseId, event.id);
		if (event.type === "tool-end" && asked.has(event.toolUseId))
			await run.answered(asked.get(event.toolUseId) as string);
		run.emit(event);
		await new Promise((resolve) => setTimeout(resolve, 1));
	}
};

const read = async (stream: ReadableStream<LanguageModelV4StreamPart>) => {
	const parts: LanguageModelV4StreamPart[] = [];
	for await (const part of stream) parts.push(part);
	return parts;
};

const user = (text: string): LanguageModelV4Message => ({
	role: "user",
	content: [{ type: "text", text }],
});

/** The parts of a step, as the assistant message a caller keeps of it. */
const toMessage = (
	parts: LanguageModelV4StreamPart[],
): LanguageModelV4Message => {
	const content: Extract<
		LanguageModelV4Message,
		{ role: "assistant" }
	>["content"] = [];
	const texts = new Map<
		string,
		{ type: "text"; text: string; providerOptions?: any }
	>();
	for (const part of parts) {
		if (part.type === "text-start") {
			const text = { type: "text" as const, text: "" };
			texts.set(part.id, text);
			content.push(text);
		} else if (part.type === "text-delta") {
			const text = texts.get(part.id);
			if (text) text.text += part.delta;
		} else if (part.type === "text-end") {
			const text = texts.get(part.id);
			if (text) text.providerOptions = part.providerMetadata;
		} else if (part.type === "tool-call")
			content.push({
				type: "tool-call",
				toolCallId: part.toolCallId,
				toolName: part.toolName,
				input: JSON.parse(part.input),
				providerExecuted: part.providerExecuted,
				providerOptions: part.providerMetadata,
			});
		else if (part.type === "tool-result" && !part.preliminary)
			content.push({
				type: "tool-result",
				toolCallId: part.toolCallId,
				toolName: part.toolName,
				output: { type: "json", value: part.result },
				providerOptions: part.providerMetadata,
			});
		else if (part.type === "custom")
			content.push({
				type: "custom",
				kind: part.kind,
				providerOptions: part.providerMetadata,
			});
	}
	return { role: "assistant", content };
};

const finishOf = (parts: LanguageModelV4StreamPart[]) =>
	parts.find((part) => part.type === "finish") as Extract<
		LanguageModelV4StreamPart,
		{ type: "finish" }
	>;

/** A turn of text, as Claude Code streams it. */
const textTurn = (
	message: string,
	uuid: string,
	text: string,
	stop = "end_turn",
): ClaudeCodeHostEvent[] =>
	[
		{
			type: "stream_event",
			parent_tool_use_id: null,
			event: {
				type: "message_start",
				message: { id: message, model: "claude", usage: { input_tokens: 3 } },
			},
		},
		{
			type: "stream_event",
			parent_tool_use_id: null,
			event: {
				type: "content_block_start",
				index: 0,
				content_block: { type: "text", text: "" },
			},
		},
		{
			type: "stream_event",
			parent_tool_use_id: null,
			event: {
				type: "content_block_delta",
				index: 0,
				delta: { type: "text_delta", text },
			},
		},
		{
			type: "assistant",
			uuid,
			parent_tool_use_id: null,
			message: { id: message, content: [{ type: "text", text }] },
		},
		{
			type: "stream_event",
			parent_tool_use_id: null,
			event: { type: "content_block_stop", index: 0 },
		},
		{
			type: "stream_event",
			parent_tool_use_id: null,
			event: {
				type: "message_delta",
				delta: { stop_reason: stop },
				usage: { output_tokens: 5 },
			},
		},
		{
			type: "stream_event",
			parent_tool_use_id: null,
			event: { type: "message_stop" },
		},
	].map((message) => ({ type: "message", message }) as ClaudeCodeHostEvent);

/** A turn calling one tool, as Claude Code streams it. */
const toolTurn = (
	message: string,
	uuid: string,
	call: { id: string; name: string; input: unknown },
): ClaudeCodeHostEvent[] =>
	[
		{
			type: "stream_event",
			parent_tool_use_id: null,
			event: { type: "message_start", message: { id: message, usage: {} } },
		},
		{
			type: "stream_event",
			parent_tool_use_id: null,
			event: {
				type: "content_block_start",
				index: 0,
				content_block: { type: "tool_use", id: call.id, name: call.name },
			},
		},
		{
			type: "assistant",
			uuid,
			parent_tool_use_id: null,
			message: {
				id: message,
				content: [{ type: "tool_use", ...call }],
			},
		},
		{
			type: "stream_event",
			parent_tool_use_id: null,
			event: { type: "content_block_stop", index: 0 },
		},
		{
			type: "stream_event",
			parent_tool_use_id: null,
			event: { type: "message_delta", delta: { stop_reason: "tool_use" } },
		},
		{
			type: "stream_event",
			parent_tool_use_id: null,
			event: { type: "message_stop" },
		},
	].map((message) => ({ type: "message", message }) as ClaudeCodeHostEvent);

const init = (session: string): ClaudeCodeHostEvent => ({
	type: "message",
	message: { type: "system", subtype: "init", session_id: session } as never,
});

const result = (session: string): ClaudeCodeHostEvent => ({
	type: "message",
	message: {
		type: "result",
		subtype: "success",
		is_error: false,
		session_id: session,
		stop_reason: "end_turn",
		total_cost_usd: 0.01,
		num_turns: 1,
		usage: {},
	} as never,
});

/** One of Claude Code's `task_*` system messages, for task `t1`. */
const task = (
	subtype: "task_started" | "task_updated" | "task_notification",
	fields: Record<string, unknown>,
): ClaudeCodeHostEvent => ({
	type: "message",
	message: {
		type: "system",
		subtype,
		task_id: "t1",
		session_id: "s1",
		...fields,
	} as never,
});

const userResult = (
	uuid: string,
	id: string,
	content: unknown,
	isError = false,
): ClaudeCodeHostEvent => ({
	type: "message",
	message: {
		type: "user",
		uuid,
		parent_tool_use_id: null,
		message: {
			role: "user",
			content: [
				{ type: "tool_result", tool_use_id: id, content, is_error: isError },
			],
		},
		tool_use_result: { stdout: content },
	} as never,
});

const call = (
	options: Partial<LanguageModelV4CallOptions> & {
		prompt: LanguageModelV4Prompt;
	},
): LanguageModelV4CallOptions => options;

const weather = {
	type: "function" as const,
	name: "get_weather",
	description: "Weather for a city",
	inputSchema: {
		type: "object" as const,
		properties: { city: { type: "string" as const } },
		required: ["city"],
	},
};

describe("claude/provider", () => {
	it("starts a session with the conversation so far, and streams its turn", async () => {
		const { host, runs } = fakeHost();
		const model = createClaudeCode({ host })("haiku");
		const { stream } = await model.doStream(
			call({
				prompt: [
					{ role: "system", content: "Be terse." },
					user("Hi"),
					{ role: "assistant", content: [{ type: "text", text: "Hello!" }] },
					user("What is 2+2?"),
				],
				temperature: 0.5,
				reasoning: "high",
			}),
		);
		const run = runs[0];
		run.emit(init("s1"), ...textTurn("m1", "u1", "4"), result("s1"));
		const parts = await read(stream);

		expect(run.request.options).toMatchObject({
			model: "haiku",
			effort: "high",
			systemPrompt: "Be terse.",
			tools: [],
			settingSources: [],
			disallowedTools: ["AskUserQuestion"],
		});
		const content = run.request.prompt.message.content as any[];
		expect(content[0].text).toContain("<conversation_history>");
		expect(content[0].text).toContain("## Assistant\n\nHello!");
		expect(content[1]).toEqual({ type: "text", text: "What is 2+2?" });

		expect(parts[0]).toEqual({
			type: "stream-start",
			warnings: [{ type: "unsupported", feature: "temperature" }],
		});
		expect(parts.filter((p) => p.type === "text-delta")).toEqual([
			{ type: "text-delta", id: "m1:0", delta: "4" },
		]);
		expect(parts.find((p) => p.type === "text-end")).toMatchObject({
			providerMetadata: { claudeCode: { sessionId: "s1", cursor: "u1" } },
		});
		expect(parts.find((p) => p.type === "custom")).toEqual({
			type: "custom",
			kind: "claude-code.step",
			providerMetadata: {
				claudeCode: { sessionId: "s1", cursor: "u1", awaiting: [] },
			},
		});
		expect(finishOf(parts)).toMatchObject({
			finishReason: { unified: "stop", raw: "end_turn" },
			usage: { inputTokens: { total: 3 }, outputTokens: { total: 5 } },
			providerMetadata: { claudeCode: { costUsd: 0.01 } },
		});
	});

	it("takes the system prompt preset and built-in tools separately", async () => {
		const { host, runs } = fakeHost();
		const claudeCode = createClaudeCode({ host });
		await claudeCode("opus", {
			systemPromptPreset: "claude_code",
			settingSources: ["user", "project"],
		}).doStream(
			call({
				prompt: [{ role: "system", content: "Extra." }, user("Hi")],
				tools: [
					{ type: "provider", id: "claude-code.Bash", name: "shell", args: {} },
					{
						type: "provider",
						id: "claude-code.AskUserQuestion",
						name: "ask",
						args: {},
					},
				],
				providerOptions: { claudeCode: { builtinTools: ["Read"] } },
			}),
		);
		expect(runs[0].request.options).toMatchObject({
			systemPrompt: { type: "preset", preset: "claude_code", append: "Extra." },
			tools: ["Read", "Bash", "AskUserQuestion"],
			settingSources: ["user", "project"],
			disallowedTools: [],
		});
	});

	it("ends a step on the caller's tool calls, and goes on with their results", async () => {
		const { host, runs } = fakeHost();
		const model = createClaudeCode({ host })("haiku");
		const prompt: LanguageModelV4Prompt = [
			user("Weather in Paris, and echo hi"),
		];
		const first = await model.doStream(call({ prompt, tools: [weather] }));
		const run = runs[0];
		expect(run.request.tools).toEqual([
			{
				name: "get_weather",
				description: "Weather for a city",
				inputSchema: weather.inputSchema,
			},
		]);
		const playing = replay(run, fixture as ClaudeCodeHostEvent[]);

		const step1 = await read(first.stream);
		const calls = step1.filter((p) => p.type === "tool-call");
		expect(calls).toMatchObject([
			{ toolName: "get_weather", input: '{"city":"Paris"}' },
			{ toolName: "Bash", providerExecuted: true, dynamic: true },
		]);
		expect(calls[0]).not.toHaveProperty("providerExecuted");
		const finish1 = finishOf(step1);
		expect(finish1.finishReason.unified).toBe("tool-calls");
		expect(finish1.providerMetadata?.claudeCode).toMatchObject({
			awaiting: ["tools"],
		});

		// The caller runs its tool and calls again.
		const weatherCall = calls[0] as Extract<
			LanguageModelV4StreamPart,
			{ type: "tool-call" }
		>;
		const second = await model.doStream(
			call({
				prompt: [
					...prompt,
					toMessage(step1),
					{
						role: "tool",
						content: [
							{
								type: "tool-result",
								toolCallId: weatherCall.toolCallId,
								toolName: "get_weather",
								output: { type: "text", value: "Rainy in Paris" },
							},
						],
					},
				],
				tools: [weather],
			}),
		);
		const step2 = await read(second.stream);
		await playing;

		expect(runs).toHaveLength(1);
		expect(run.answers.get("q-mcp")).toEqual({
			type: "tool-result",
			content: [{ type: "text", text: "Rainy in Paris" }],
			isError: false,
		});
		expect(step2.find((p) => p.type === "tool-result")).toMatchObject({
			toolName: "Bash",
			result: "hi",
			dynamic: true,
			providerMetadata: {
				claudeCode: { output: { stdout: "hi", stderr: "" } },
			},
		});
		const text = step2
			.filter((p) => p.type === "text-delta")
			.map((p) => (p as { delta: string }).delta)
			.join("");
		expect(text).toBe('Paris is sunny, and `echo hi` printed "hi".');
		expect(finishOf(step2).finishReason.unified).toBe("stop");
	});

	it("asks for approval in the same step as the call, again if it must", async () => {
		const { host, runs } = fakeHost();
		const model = createClaudeCode({ host })("haiku");
		const prompt: LanguageModelV4Prompt = [user("Delete it")];
		const first = await model.doStream(call({ prompt }));
		const run = runs[0];
		run.emit(
			init("s1"),
			...toolTurn("m1", "u1", {
				id: "t1",
				name: "Bash",
				input: { command: "rm x" },
			}),
		);
		// Asked for after the turn ended, as Claude Code does.
		setTimeout(
			() =>
				run.emit(
					{ type: "tool-start", toolUseId: "t1", name: "Bash" },
					{
						type: "permission",
						id: "p1",
						toolUseId: "t1",
						name: "Bash",
						input: { command: "rm x" },
						title: "Claude wants to run rm x",
					},
				),
			5,
		);
		const step1 = await read(first.stream);
		expect(step1.map((p) => p.type)).toContain("tool-approval-request");
		expect(step1.find((p) => p.type === "tool-approval-request")).toMatchObject(
			{
				approvalId: "t1",
				toolCallId: "t1",
				providerMetadata: { claudeCode: { title: "Claude wants to run rm x" } },
			},
		);
		expect(finishOf(step1).providerMetadata?.claudeCode).toMatchObject({
			awaiting: ["approval"],
		});

		const second = await model.doStream(
			call({
				prompt: [
					...prompt,
					toMessage(step1),
					{
						role: "tool",
						content: [
							{
								type: "tool-approval-response",
								approvalId: "t1",
								approved: true,
							},
						],
					},
				],
			}),
		);
		expect(run.answers.get("p1")).toEqual({
			type: "permission",
			result: { behavior: "allow", updatedInput: { command: "rm x" } },
		});
		run.emit(
			{ type: "tool-end", toolUseId: "t1" },
			userResult("u2", "t1", "removed"),
			...textTurn("m2", "u3", "Done."),
			result("s1"),
		);
		const step2 = await read(second.stream);
		expect(step2.find((p) => p.type === "tool-result")).toMatchObject({
			toolCallId: "t1",
			result: "removed",
			providerMetadata: { claudeCode: { sessionId: "s1", cursor: "u2" } },
		});
		expect(finishOf(step2).finishReason.unified).toBe("stop");
	});

	it("streams a call again with its approval when it was asked about later", async () => {
		const { host, runs } = fakeHost();
		const model = createClaudeCode({ host })("haiku");
		const prompt: LanguageModelV4Prompt = [user("Go")];
		const first = await model.doStream(call({ prompt, tools: [weather] }));
		const run = runs[0];
		// The caller's call, then a built-in queued behind it.
		run.emit(
			init("s1"),
			...toolTurn("m1", "u1", {
				id: "c1",
				name: "mcp__ai-sdk__get_weather",
				input: { city: "Oslo" },
			}),
			{
				type: "tool-call",
				id: "q1",
				toolUseId: "c1",
				name: "get_weather",
				input: { city: "Oslo" },
			},
		);
		const step1 = await read(first.stream);
		expect(finishOf(step1).finishReason.unified).toBe("tool-calls");

		const second = await model.doStream(
			call({
				prompt: [
					...prompt,
					toMessage(step1),
					{
						role: "tool",
						content: [
							{
								type: "tool-result",
								toolCallId: "c1",
								toolName: "get_weather",
								output: { type: "text", value: "Cold" },
							},
						],
					},
				],
				tools: [weather],
			}),
		);
		run.emit(
			userResult("u2", "c1", "Cold"),
			...toolTurn("m2", "u3", {
				id: "t2",
				name: "Write",
				input: { file_path: "a" },
			}),
		);
		setTimeout(
			() =>
				run.emit({
					type: "permission",
					id: "p2",
					toolUseId: "t2",
					name: "Write",
					input: { file_path: "a" },
				}),
			5,
		);
		const step2 = await read(second.stream);
		expect(
			step2
				.filter(
					(p) => p.type === "tool-call" || p.type === "tool-approval-request",
				)
				.map((p) => p.type),
		).toEqual(["tool-call", "tool-approval-request"]);
	});

	it("routes AskUserQuestion to the caller and answers with what it returned", async () => {
		const { host, runs } = fakeHost();
		const claudeCode = createClaudeCode({ host });
		const model = claudeCode("haiku");
		const tools = [
			{
				type: "provider" as const,
				id: "claude-code.AskUserQuestion" as const,
				name: "ask",
				args: {},
			},
		];
		const prompt: LanguageModelV4Prompt = [user("Pick for me")];
		const input = { questions: [{ question: "Which?", header: "H" }] };
		const first = await model.doStream(call({ prompt, tools }));
		const run = runs[0];
		run.emit(
			init("s1"),
			...toolTurn("m1", "u1", { id: "a1", name: "AskUserQuestion", input }),
		);
		setTimeout(
			() =>
				run.emit({
					type: "permission",
					id: "p1",
					toolUseId: "a1",
					name: "AskUserQuestion",
					input,
				}),
			5,
		);
		const step1 = await read(first.stream);
		const asked = step1.find((p) => p.type === "tool-call");
		expect(asked).toMatchObject({ toolName: "ask" });
		expect(asked).not.toHaveProperty("providerExecuted");
		expect(step1.some((p) => p.type === "tool-approval-request")).toBe(false);

		await model.doStream(
			call({
				prompt: [
					...prompt,
					toMessage(step1),
					{
						role: "tool",
						content: [
							{
								type: "tool-result",
								toolCallId: "a1",
								toolName: "ask",
								output: { type: "json", value: { answers: { "Which?": "A" } } },
							},
						],
					},
				],
				tools,
			}),
		);
		expect(run.answers.get("p1")).toEqual({
			type: "permission",
			result: {
				behavior: "allow",
				updatedInput: { ...input, answers: { "Which?": "A" } },
			},
		});
	});

	it("sends what the user said meanwhile into the live session", async () => {
		const { host, runs } = fakeHost();
		const model = createClaudeCode({ host })("haiku");
		const prompt: LanguageModelV4Prompt = [user("Hi")];
		const first = await model.doStream(call({ prompt }));
		runs[0].emit(init("s1"), ...textTurn("m1", "u1", "Hello"), result("s1"));
		const step1 = await read(first.stream);

		const second = await model.doStream(
			call({ prompt: [...prompt, toMessage(step1), user("And you?")] }),
		);
		expect(runs).toHaveLength(1);
		expect(runs[0].sent).toEqual([
			{
				type: "user",
				message: {
					role: "user",
					content: [{ type: "text", text: "And you?" }],
				},
				parent_tool_use_id: null,
				priority: "next",
			},
		]);
		runs[0].emit(...textTurn("m2", "u2", "Fine"), result("s1"));
		const step2 = await read(second.stream);
		expect(finishOf(step2).providerMetadata?.claudeCode).toMatchObject({
			sessionId: "s1",
			cursor: "u2",
		});
	});

	it("continues from any place the last step left, without its marker", async () => {
		const { host, runs } = fakeHost();
		const model = createClaudeCode({ host })("haiku");
		const prompt: LanguageModelV4Prompt = [user("Go")];
		const first = await model.doStream(call({ prompt, tools: [weather] }));
		const run = runs[0];
		run.emit(
			init("s1"),
			...toolTurn("m1", "u1", {
				id: "c1",
				name: "mcp__ai-sdk__get_weather",
				input: { city: "Oslo" },
			}),
		);
		const step1 = await read(first.stream);
		// Kept without the step's marker, as a caller with its own storage may.
		const kept = toMessage(step1);

		// TODO - diagnose types
		// kept.content = kept.content.filter((part) => part.type !== "custom");

		const content =
			typeof kept.content === "string"
				? [{ type: "text", text: kept.content }]
				: kept.content;
		Object.assign(kept, {
			content: content.filter((part) => part.type !== "custom"),
		});

		const second = await model.doStream(
			call({
				prompt: [
					...prompt,
					kept,
					{
						role: "tool",
						content: [
							{
								type: "tool-result",
								toolCallId: "c1",
								toolName: "get_weather",
								// Answers and results may come back as fenced JSON text.
								output: { type: "text", value: "Cold" },
							},
						],
					},
				],
				tools: [weather],
			}),
		);
		// The caller's result is saved without a part: the cursor moves on.
		run.emit(
			userResult("u2", "c1", "Cold"),
			...textTurn("m2", "u3", "Brr"),
			result("s1"),
		);
		const step2 = await read(second.stream);
		expect(runs).toHaveLength(1);
		expect(finishOf(step2).providerMetadata?.claudeCode).toMatchObject({
			route: "continue",
		});

		const third = await model.doStream(
			call({ prompt: [...prompt, kept, toMessage(step2), user("More")] }),
		);
		expect(runs).toHaveLength(1);
		expect(runs[0].sent).toHaveLength(1);
		run.emit(...textTurn("m3", "u4", "Ok"), result("s1"));
		await read(third.stream);
	});

	it("reads answers given as fenced JSON text", async () => {
		const { host, runs } = fakeHost();
		const model = createClaudeCode({ host })("haiku");
		const tools = [
			{
				type: "provider" as const,
				id: "claude-code.AskUserQuestion" as const,
				name: "AskUserQuestion",
				args: {},
			},
		];
		const prompt: LanguageModelV4Prompt = [user("Ask me")];
		const input = { questions: [{ question: "Tea?" }] };
		const first = await model.doStream(call({ prompt, tools }));
		runs[0].emit(
			init("s1"),
			...toolTurn("m1", "u1", { id: "a1", name: "AskUserQuestion", input }),
			{
				type: "permission",
				id: "p1",
				toolUseId: "a1",
				name: "AskUserQuestion",
				input,
			},
		);
		const step1 = await read(first.stream);
		await model.doStream(
			call({
				prompt: [
					...prompt,
					toMessage(step1),
					{
						role: "tool",
						content: [
							{
								type: "tool-result",
								toolCallId: "a1",
								toolName: "AskUserQuestion",
								output: {
									type: "content",
									value: [
										{
											type: "text",
											text: '```json\n{"answers":{"Tea?":"Yes"}}\n```',
										},
									],
								},
							},
						],
					},
				],
				tools,
			}),
		);
		expect(runs[0].answers.get("p1")).toMatchObject({
			result: {
				behavior: "allow",
				updatedInput: { answers: { "Tea?": "Yes" } },
			},
		});
	});

	it("forks a saved session at the cursor, answering what it left waiting", async () => {
		const { host, runs } = fakeHost(["old"]);
		const model = createClaudeCode({ host })("haiku");
		await model.doStream(
			call({
				prompt: [
					user("Weather?"),
					{
						role: "assistant",
						content: [
							{
								type: "tool-call",
								toolCallId: "c1",
								toolName: "get_weather",
								input: { city: "Rome" },
								providerOptions: {
									claudeCode: { sessionId: "old", cursor: "u9" },
								},
							},
							{
								type: "tool-call",
								toolCallId: "t1",
								toolName: "Bash",
								input: { command: "ls" },
								providerExecuted: true,
								providerOptions: {
									claudeCode: { sessionId: "old", cursor: "u9" },
								},
							},
						],
					},
					{
						role: "tool",
						content: [
							{
								type: "tool-result",
								toolCallId: "c1",
								toolName: "get_weather",
								output: { type: "text", value: "Warm" },
							},
						],
					},
					user("Thanks"),
				],
				tools: [weather],
			}),
		);
		expect(runs[0].request.options).toMatchObject({
			resume: "old",
			resumeSessionAt: "u9",
			forkSession: true,
		});
		expect(runs[0].request.prompt.message.content).toEqual([
			{
				type: "tool_result",
				tool_use_id: "c1",
				content: [{ type: "text", text: "Warm" }],
			},
			{
				type: "tool_result",
				tool_use_id: "t1",
				content: "This call did not run: the session restarted.",
				is_error: true,
			},
			{ type: "text", text: "Thanks" },
		]);
	});

	it("starts over with a transcript when the session isn't saved", async () => {
		const { host, runs } = fakeHost();
		const model = createClaudeCode({ host })("haiku");
		await model.doStream(
			call({
				prompt: [
					user("Hi"),
					{
						role: "assistant",
						content: [
							{
								type: "text",
								text: "Hey",
								providerOptions: {
									claudeCode: { sessionId: "gone", cursor: "u1" },
								},
							},
						],
					},
					user("Again"),
				],
			}),
		);
		expect(runs[0].request.options).not.toHaveProperty("resume");
		const content = runs[0].request.prompt.message.content as any[];
		expect(content[0].text).toContain("## Assistant\n\nHey");
		expect(content[1]).toEqual({ type: "text", text: "Again" });
	});

	it("streams a subagent's work as its call's preliminary results", async () => {
		const { host, runs } = fakeHost();
		const model = createClaudeCode({ host })("haiku");
		const first = await model.doStream(call({ prompt: [user("Research")] }));
		const run = runs[0];
		run.emit(
			init("s1"),
			...toolTurn("m1", "u1", {
				id: "ag",
				name: "Agent",
				input: { prompt: "x" },
			}),
			{ type: "tool-start", toolUseId: "ag", name: "Agent" },
			{
				type: "message",
				message: {
					type: "assistant",
					uuid: "sub1",
					parent_tool_use_id: "ag",
					message: { id: "sm1", content: [{ type: "text", text: "Looking" }] },
				} as never,
			},
			{ type: "tool-end", toolUseId: "ag" },
			userResult("u2", "ag", [{ type: "text", text: "Found it" }]),
		);
		const parts = await read(first.stream);
		const results = parts.filter((p) => p.type === "tool-result");
		expect(results).toMatchObject([
			{
				toolCallId: "ag",
				preliminary: true,
				result: {
					type: "subagent",
					messages: [
						{ role: "assistant", content: [{ type: "text", text: "Looking" }] },
					],
				},
			},
			{ toolCallId: "ag", result: [{ type: "text", text: "Found it" }] },
		]);
		expect(finishOf(parts).providerMetadata?.claudeCode).toMatchObject({
			awaiting: ["continue"],
		});
	});

	it("reports background work finishing, then streams the turn it brings", async () => {
		const { host, runs } = fakeHost();
		const model = createClaudeCode({ host })("haiku");
		const prompt: LanguageModelV4Prompt = [user("Research")];
		const first = await model.doStream(call({ prompt }));
		const run = runs[0];
		run.emit(
			init("s1"),
			...toolTurn("m1", "u1", { id: "ag", name: "Agent", input: {} }),
			task("task_started", { tool_use_id: "ag", is_backgrounded: true }),
			userResult("u2", "ag", [{ type: "text", text: "Launched" }]),
			...textTurn("m2", "u3", "Waiting on it."),
			result("s1"),
		);
		const launched = await read(first.stream);
		prompt.push(toMessage(launched));
		const waiting = await read((await model.doStream(call({ prompt }))).stream);
		expect(finishOf(waiting).providerMetadata?.claudeCode).toMatchObject({
			awaiting: ["background"],
		});
		prompt.push(toMessage(waiting));

		const second = await model.doStream(call({ prompt }));
		run.emit(
			task("task_notification", {
				tool_use_id: "ag",
				status: "completed",
				summary: "Found it",
			}),
		);
		// Claude Code takes a moment to start the turn the notification brings.
		setTimeout(
			() => run.emit(...textTurn("m3", "u4", "It found it."), result("s1")),
			20,
		);
		const next = await read(second.stream);
		expect(next.filter((p) => p.type === "custom")[0]).toMatchObject({
			kind: "claude-code.task-notification",
			providerMetadata: {
				claudeCode: {
					toolCallId: "ag",
					toolName: "Agent",
					summary: "Found it",
				},
			},
		});
		expect(next).toContainEqual(
			expect.objectContaining({ type: "text-delta", delta: "It found it." }),
		);
	});

	it("gives a subagent's handed-back report as its call's result", async () => {
		const { host, runs } = fakeHost();
		const model = createClaudeCode({ host })("haiku");
		const { stream } = await model.doStream(call({ prompt: [user("Look")] }));
		runs[0].emit(
			init("s1"),
			...toolTurn("m1", "u1", { id: "ag", name: "Agent", input: {} }),
			{
				type: "message",
				message: {
					type: "assistant",
					uuid: "sub1",
					parent_tool_use_id: "ag",
					message: {
						id: "sm1",
						content: [
							{
								type: "tool_use",
								id: "hb",
								name: "SubagentHandback",
								input: { message: "The pages 404." },
							},
						],
					},
				} as never,
			},
			userResult("u2", "ag", [
				{ type: "text", text: "This agent's report was delivered to you." },
			]),
			...textTurn("m2", "u3", "It failed."),
			result("s1"),
		);
		const parts = await read(stream);
		expect(parts).toContainEqual(
			expect.objectContaining({
				type: "tool-result",
				toolCallId: "ag",
				result: [{ type: "text", text: "The pages 404." }],
			}),
		);
	});

	it("goes on after a result for a notification no query took in", async () => {
		const { host, runs } = fakeHost();
		const model = createClaudeCode({ host })("haiku");
		const prompt: LanguageModelV4Prompt = [user("Run it in the background")];
		const first = await model.doStream(call({ prompt }));
		const run = runs[0];
		run.emit(
			init("s1"),
			...toolTurn("m1", "u1", { id: "sh", name: "Bash", input: {} }),
			task("task_started", { tool_use_id: "sh", is_backgrounded: true }),
			userResult("u2", "sh", "Running in the background."),
		);
		prompt.push(toMessage(await read(first.stream)));

		const second = await model.doStream(call({ prompt }));
		// It ends while the model is still saying it waits.
		run.emit(
			task("task_notification", { tool_use_id: "sh", status: "completed" }),
			...textTurn("m2", "u3", "Still running."),
			result("s1"),
		);
		// Then Claude Code takes it in with a query of its own.
		setTimeout(
			() =>
				run.emit(
					init("s1"),
					...textTurn("m3", "u4", "It printed a.txt."),
					result("s1"),
				),
			20,
		);
		const waiting = await read(second.stream);
		expect(finishOf(waiting).providerMetadata?.claudeCode).toMatchObject({
			awaiting: ["background"],
		});
		prompt.push(toMessage(waiting));

		const third = await read((await model.doStream(call({ prompt }))).stream);
		expect(third).toContainEqual(
			expect.objectContaining({
				type: "text-delta",
				delta: "It printed a.txt.",
			}),
		);
		expect(finishOf(third).providerMetadata?.claudeCode).toMatchObject({
			awaiting: [],
		});
	});

	it("ends at a result when the query took the notification in", async () => {
		const { host, runs } = fakeHost();
		const model = createClaudeCode({ host })("haiku");
		const { stream } = await model.doStream(call({ prompt: [user("Go")] }));
		runs[0].emit(
			init("s1"),
			task("task_started", { tool_use_id: "sh", is_backgrounded: true }),
			task("task_notification", { tool_use_id: "sh", status: "completed" }),
			...textTurn("m1", "u1", "It finished."),
			result("s1"),
		);
		const parts = await read(stream);
		expect(finishOf(parts).providerMetadata?.claudeCode).toMatchObject({
			awaiting: [],
		});
	});

	it("leaves a foreground task's notification to its call's result", async () => {
		const { host, runs } = fakeHost();
		const model = createClaudeCode({ host })("haiku");
		const { stream } = await model.doStream(call({ prompt: [user("Run")] }));
		runs[0].emit(
			init("s1"),
			...toolTurn("m1", "u1", { id: "sh", name: "Bash", input: {} }),
			task("task_started", { tool_use_id: "sh", is_backgrounded: false }),
			task("task_notification", {
				tool_use_id: "sh",
				status: "completed",
				summary: "List files",
			}),
			userResult("u2", "sh", "a.txt"),
			...textTurn("m2", "u3", "One file."),
			result("s1"),
		);
		const parts = await read(stream);
		expect(
			parts.filter(
				(p) =>
					p.type === "custom" && p.kind === "claude-code.task-notification",
			),
		).toEqual([]);
		expect(parts).toContainEqual(
			expect.objectContaining({
				type: "tool-result",
				toolCallId: "sh",
				result: "a.txt",
			}),
		);
	});

	it("interrupts the turn when the call is aborted", async () => {
		const { host, runs } = fakeHost();
		const model = createClaudeCode({ host })("haiku");
		const abort = new AbortController();
		const { stream } = await model.doStream(
			call({ prompt: [user("Long task")], abortSignal: abort.signal }),
		);
		runs[0].emit(init("s1"));
		setTimeout(() => abort.abort(), 5);
		const parts = await read(stream);
		expect(runs[0].interrupted).toBe(1);
		expect(finishOf(parts).finishReason.unified).toBe("other");
	});

	it("aggregates a step in doGenerate", async () => {
		const { host, runs } = fakeHost();
		const model = createClaudeCode({ host })("haiku");
		const generating = model.doGenerate(call({ prompt: [user("Hi")] }));
		await new Promise((resolve) => setTimeout(resolve, 1));
		runs[0].emit(init("s1"), ...textTurn("m1", "u1", "Hello"), result("s1"));
		const { content, finishReason } = await generating;
		expect(content[0]).toMatchObject({ type: "text", text: "Hello" });
		expect(finishReason.unified).toBe("stop");
	});
});
