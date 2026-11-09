import type {
	LanguageModelV4CallOptions,
	LanguageModelV4Prompt,
	LanguageModelV4StreamPart,
} from "@ai-sdk/provider";
import type {
	CursorHost,
	CursorHostEvent,
	CursorOpenRequest,
	CursorSendRequest,
	CursorToolAnswer,
} from "#cursor/host/protocol.ts";
import { modelParams } from "#cursor/model.ts";
import { createCursor } from "#cursor/provider.ts";

/** A host that records what it is asked and emits what a test says. */
const fakeHost = (
	options: { lastRun?: { id: string; status: string } } = {},
) => {
	const opened: CursorOpenRequest[] = [];
	const sent: { agentId: string; request: CursorSendRequest }[] = [];
	const answers: { toolCallId: string; answer: CursorToolAnswer }[] = [];
	const steered: string[] = [];
	const cancelled: string[] = [];
	const closed: string[] = [];
	let handler: ((event: CursorHostEvent) => void) | undefined;
	let agents = 0;
	let runs = 0;
	const host: CursorHost = {
		status: async () => ({
			loggedIn: true,
			models: [
				{
					id: "claude-opus-5-5",
					parameters: [
						{
							id: "effort",
							values: ["low", "medium", "high", "xhigh", "max"].map(
								(value) => ({ value }),
							),
						},
					],
				},
			],
		}),
		async open(request) {
			opened.push(request);
			return request.agentId ?? `agent-${++agents}`;
		},
		async send(agentId, request, onEvent) {
			sent.push({ agentId, request });
			handler = onEvent;
			return `run-${++runs}`;
		},
		answer: async (_agentId, toolCallId, answer) => {
			answers.push({ toolCallId, answer });
		},
		steer: async (_agentId, text) => {
			steered.push(text);
			return true;
		},
		cancel: async (agentId) => {
			cancelled.push(agentId);
		},
		close: async (agentId) => {
			closed.push(agentId);
		},
		lastRun: async () => options.lastRun,
	};
	const emit = (...events: CursorHostEvent[]) => {
		for (const event of events) handler?.(event);
	};
	return { host, opened, sent, answers, steered, cancelled, closed, emit };
};

const delta = (update: any): CursorHostEvent => ({ type: "delta", update });
const text = (value: string) => delta({ type: "text-delta", text: value });
const thinking = (value: string) =>
	delta({ type: "thinking-delta", text: value });
const listed = delta({ type: "tool-requests-listed", callCount: 1 });
const clientCall = (toolName: string, args: unknown) => ({
	type: "mcp",
	args: { providerIdentifier: "custom-user-tools", toolName, args },
});
const started = (callId: string, toolCall: unknown) =>
	delta({ type: "tool-call-started", callId, toolCall, modelCallId: "m" });
const completed = (callId: string, toolCall: unknown) =>
	delta({ type: "tool-call-completed", callId, toolCall, modelCallId: "m" });
const usage = {
	inputTokens: 100,
	outputTokens: 10,
	cacheReadTokens: 60,
	cacheWriteTokens: 0,
};
const ended: CursorHostEvent[] = [
	delta({ type: "turn-ended", usage }),
	{ type: "result", status: "finished", usage, requestId: "req-1" },
	{ type: "end" },
];

const read = async (stream: ReadableStream<LanguageModelV4StreamPart>) => {
	const parts: LanguageModelV4StreamPart[] = [];
	for await (const part of stream) parts.push(part);
	return parts;
};

const finishOf = (parts: LanguageModelV4StreamPart[]) =>
	parts.find((part) => part.type === "finish") as Extract<
		LanguageModelV4StreamPart,
		{ type: "finish" }
	>;

const user = (value: string): LanguageModelV4Prompt[number] => ({
	role: "user",
	content: [{ type: "text", text: value }],
});

const weather = {
	type: "function" as const,
	name: "get_weather",
	description: "The weather",
	inputSchema: {
		type: "object" as const,
		properties: { city: { type: "string" as const } },
	},
};

const call = (
	prompt: LanguageModelV4Prompt,
	extra: Partial<LanguageModelV4CallOptions> = {},
): LanguageModelV4CallOptions => ({ prompt, ...extra });

describe("cursor provider", () => {
	it("opens an agent with no built-ins, settings or prompt of its own, and streams a reply", async () => {
		const fake = fakeHost();
		const model = createCursor({ host: fake.host })("composer-2.5");
		const { stream } = await model.doStream(
			call([{ role: "system", content: "Be terse." }, user("Hi")]),
		);
		fake.emit(thinking("Hm"), text("Hello"), text("!"), ...ended);
		const parts = await read(stream);

		expect(fake.opened[0].options).toEqual({
			model: { id: "composer-2.5" },
			tools: [],
		});
		expect(fake.sent[0].request.message.text).toBe(
			"<system_instructions>\nBe terse.\n</system_instructions>\n\nHi",
		);
		const types = parts.map((part) => part.type);
		expect(types).toEqual([
			"stream-start",
			"response-metadata",
			"reasoning-start",
			"reasoning-delta",
			"reasoning-end",
			"text-start",
			"text-delta",
			"text-delta",
			"text-end",
			"custom",
			"finish",
		]);
		const finish = finishOf(parts);
		expect(finish.finishReason.unified).toBe("stop");
		expect(finish.usage.inputTokens.total).toBe(160);
		expect(finish.usage.inputTokens.cacheRead).toBe(60);
		expect(finish.usage.outputTokens.total).toBe(10);
		expect(finish.providerMetadata?.cursor).toMatchObject({
			agentId: "agent-1",
			cursor: "run-1#1",
			route: "new",
			requestId: "req-1",
		});
	});

	it("replaces Cursor's prompt only when asked to", async () => {
		const fake = fakeHost();
		const model = createCursor({ host: fake.host })("composer-2.5", {
			systemPromptPreset: null,
		});
		const { stream } = await model.doStream(
			call([{ role: "system", content: "Be terse." }, user("Hi")]),
		);
		fake.emit(...ended);
		await read(stream);
		expect(fake.opened[0].options.systemPrompt).toBe("Be terse.");
		expect(fake.sent[0].request.message.text).toBe("Hi");
	});

	it("hands the caller's tool calls back, one step per call, and answers them from the next", async () => {
		const fake = fakeHost();
		const provider = createCursor({ host: fake.host });
		const prompt: LanguageModelV4Prompt = [user("Weather in Paris?")];
		const first = await provider("composer-2.5").doStream(
			call(prompt, { tools: [weather] }),
		);
		const toolCall = clientCall("get_weather", { city: "Paris" });
		fake.emit(
			text("Checking."),
			started("c1", toolCall),
			{
				type: "tool-call",
				toolCallId: "c1",
				name: "get_weather",
				input: { city: "Paris" },
			},
			listed,
		);
		const parts = await read(first.stream);

		expect(fake.opened[0].options.tools).toEqual(["mcp"]);
		expect(fake.sent[0].request.tools).toEqual([
			{
				name: "get_weather",
				description: "The weather",
				inputSchema: weather.inputSchema,
			},
		]);
		const callPart = parts.find((part) => part.type === "tool-call") as any;
		expect(callPart).toMatchObject({
			toolCallId: "c1",
			toolName: "get_weather",
			input: '{"city":"Paris"}',
		});
		expect(callPart.providerExecuted).toBeUndefined();
		const finish = finishOf(parts);
		expect(finish.finishReason.unified).toBe("tool-calls");
		expect(finish.providerMetadata?.cursor?.awaiting).toEqual(["tools"]);

		const second = await provider("composer-2.5").doStream(
			call(
				[
					...prompt,
					{
						role: "assistant",
						content: [
							{ type: "text", text: "Checking." },
							{
								type: "tool-call",
								toolCallId: "c1",
								toolName: "get_weather",
								input: { city: "Paris" },
								providerOptions: { cursor: callPart.providerMetadata.cursor },
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
								output: { type: "text", value: "Sunny" },
							},
						],
					},
				],
				{ tools: [weather] },
			),
		);
		expect(fake.answers).toEqual([
			{
				toolCallId: "c1",
				answer: { content: [{ type: "text", text: "Sunny" }] },
			},
		]);
		fake.emit(completed("c1", toolCall), text("It's sunny."), ...ended);
		const rest = await read(second.stream);
		expect(fake.opened).toHaveLength(1);
		expect(fake.sent).toHaveLength(1);
		expect(rest.filter((part) => part.type === "tool-result")).toEqual([]);
		expect(finishOf(rest).finishReason.unified).toBe("stop");
		expect(finishOf(rest).providerMetadata?.cursor?.route).toBe("continue");
	});

	it("hands back serial client calls while a built-in is open, without waiting for the announced count", async () => {
		const fake = fakeHost();
		const provider = createCursor({ host: fake.host });
		const prompt: LanguageModelV4Prompt = [user("Weather and read a.txt")];
		const next = () =>
			provider("composer-2.5").doStream(call(prompt, { tools: [weather] }));
		const reply = (parts: LanguageModelV4StreamPart[]) => {
			const part = parts.find(
				(part) => part.type === "tool-call" && !part.providerExecuted,
			);
			if (part?.type !== "tool-call") throw new Error("Missing client call");
			prompt.push(
				{
					role: "assistant",
					content: [
						{
							type: "tool-call",
							toolCallId: part.toolCallId,
							toolName: part.toolName,
							input: JSON.parse(part.input),
							providerOptions: part.providerMetadata,
						},
					],
				},
				{
					role: "tool",
					content: [
						{
							type: "tool-result",
							toolCallId: part.toolCallId,
							toolName: part.toolName,
							output: { type: "text", value: "Sunny" },
						},
					],
				},
			);
		};
		const builtin = { type: "read", args: { path: "a.txt" } };
		const first = clientCall("get_weather", { city: "Paris" });
		const second = clientCall("get_weather", { city: "Oslo" });
		const one = await next();
		fake.emit(
			delta({ type: "tool-requests-listed", callCount: 4 }),
			started("r1", builtin),
			started("c1", first),
			{
				type: "tool-call",
				toolCallId: "c1",
				name: "get_weather",
				input: { city: "Paris" },
			},
		);
		const parts = await read(one.stream);
		expect(parts.filter((part) => part.type === "tool-result")).toHaveLength(0);
		expect(finishOf(parts).providerMetadata?.cursor?.awaiting).toEqual([
			"tools",
		]);
		reply(parts);
		const two = await next();
		// Cursor starts this only after the first answer, with no new listing.
		fake.emit(started("c2", second), {
			type: "tool-call",
			toolCallId: "c2",
			name: "get_weather",
			input: { city: "Oslo" },
		});
		const more = await read(two.stream);
		expect(more.find((part) => part.type === "tool-call")).toMatchObject({
			toolCallId: "c2",
		});
		expect(finishOf(more).finishReason.unified).toBe("tool-calls");
		reply(more);
		const three = await next();
		fake.emit(
			completed("r1", {
				...builtin,
				result: { status: "success", value: { content: "hi" } },
			}),
			text("Done"),
			...ended,
		);
		const rest = await read(three.stream);
		expect(rest.find((part) => part.type === "tool-result")).toMatchObject({
			toolCallId: "r1",
		});
		expect(finishOf(rest).finishReason.unified).toBe("stop");
		expect(fake.answers.map((answer) => answer.toolCallId)).toEqual([
			"c1",
			"c2",
		]);
		await provider.close();
	});

	it("keeps an answer given before Cursor asks for it", async () => {
		const fake = fakeHost();
		const provider = createCursor({ host: fake.host });
		const prompt: LanguageModelV4Prompt = [user("Weather?")];
		const first = await provider("composer-2.5").doStream(
			call(prompt, { tools: [weather] }),
		);
		const toolCall = clientCall("get_weather", { city: "Oslo" });
		fake.emit(started("c1", toolCall), listed);
		const parts = await read(first.stream);
		const callPart = parts.find((part) => part.type === "tool-call") as any;

		await provider("composer-2.5").doStream(
			call(
				[
					...prompt,
					{
						role: "assistant",
						content: [
							{
								type: "tool-call",
								toolCallId: "c1",
								toolName: "get_weather",
								input: {},
								providerOptions: { cursor: callPart.providerMetadata.cursor },
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
								output: { type: "error-text", value: "Down" },
							},
						],
					},
				],
				{ tools: [weather] },
			),
		);
		expect(fake.answers).toEqual([]);
		fake.emit({
			type: "tool-call",
			toolCallId: "c1",
			name: "get_weather",
			input: {},
		});
		expect(fake.answers).toEqual([
			{
				toolCallId: "c1",
				answer: { content: [{ type: "text", text: "Down" }], isError: true },
			},
		]);
	});

	it("streams Cursor's own tools with their results, and goes on in the next step", async () => {
		const fake = fakeHost();
		const provider = createCursor({ host: fake.host });
		const read_ = provider.tools.read({});
		const { stream } = await provider("composer-2.5").doStream(
			call([user("Read a.txt")], {
				tools: [
					{ type: "provider", id: "cursor.read", name: "Read", args: {} },
					{ type: "provider", id: "cursor.write", name: "Write", args: {} },
				],
			}),
		);
		expect(read_).toBeDefined();
		const toolCall = { type: "read", args: { path: "a.txt" } };
		fake.emit(
			started("r1", toolCall),
			listed,
			completed("r1", {
				...toolCall,
				result: { status: "success", value: { content: "hi" } },
			}),
			text("It says hi."),
		);
		const parts = await read(stream);
		expect(fake.opened[0].options.tools).toEqual(["read", "edit"]);
		expect(parts.find((part) => part.type === "tool-call")).toMatchObject({
			toolName: "Read",
			providerExecuted: true,
		});
		expect(parts.find((part) => part.type === "tool-result")).toMatchObject({
			toolCallId: "r1",
			toolName: "Read",
			result: { status: "success", value: { content: "hi" } },
		});
		// The next model call is the next step's.
		expect(parts.some((part) => part.type === "text-delta")).toBe(false);
		const finish = finishOf(parts);
		expect(finish.finishReason.unified).toBe("tool-calls");
		expect(finish.providerMetadata?.cursor?.awaiting).toEqual(["continue"]);
	});

	it("remembers a batch announced before its built-in starts", async () => {
		const fake = fakeHost();
		const provider = createCursor({ host: fake.host });
		const { stream } = await provider("composer-2.5").doStream(
			call([user("Read a.txt")]),
		);
		const toolCall = { type: "read", args: { path: "a.txt" } };
		fake.emit(
			listed,
			started("r1", toolCall),
			completed("r1", {
				...toolCall,
				result: { status: "success", value: { content: "hi" } },
			}),
		);
		const parts = await read(stream);
		expect(parts.find((part) => part.type === "tool-result")).toMatchObject({
			toolCallId: "r1",
		});
		expect(finishOf(parts).providerMetadata?.cursor?.awaiting).toEqual([
			"continue",
		]);
		await provider.close();
	});

	it("ends no step on calls only Cursor sees", async () => {
		const fake = fakeHost();
		const { stream } = await createCursor({ host: fake.host })(
			"composer-2.5",
		).doStream(call([user("Hi")], { tools: [weather] }));
		fake.emit(listed, text("Hello"), ...ended);
		const parts = await read(stream);
		expect(parts.some((part) => part.type === "text-delta")).toBe(true);
		expect(finishOf(parts).finishReason.unified).toBe("stop");
	});

	it("cancels the run when the call is aborted", async () => {
		const fake = fakeHost();
		const abort = new AbortController();
		const { stream } = await createCursor({ host: fake.host })(
			"composer-2.5",
		).doStream(call([user("Hi")], { abortSignal: abort.signal }));
		fake.emit(text("Hel"));
		const reading = read(stream);
		await new Promise((resolve) => setTimeout(resolve, 10));
		abort.abort();
		const parts = await reading;
		expect(fake.cancelled).toEqual(["agent-1"]);
		expect(finishOf(parts).finishReason.unified).toBe("other");
	});

	it("resumes a saved agent where its last run finished, and starts over otherwise", async () => {
		const prompt = (cursor: string): LanguageModelV4Prompt => [
			user("Hi"),
			{
				role: "assistant",
				content: [
					{
						type: "text",
						text: "Hello",
						providerOptions: { cursor: { agentId: "agent-9", cursor } },
					},
				],
			},
			user("Again"),
		];
		const saved = fakeHost({ lastRun: { id: "run-7", status: "finished" } });
		const resumed = await createCursor({ host: saved.host })(
			"composer-2.5",
		).doStream(call(prompt("run-7#2")));
		expect(saved.opened[0].agentId).toBe("agent-9");
		expect(saved.sent[0].request.message.text).toBe("Again");
		expect(resumed.request?.body).toMatchObject({ route: "resume" });

		const moved = fakeHost({ lastRun: { id: "run-8", status: "finished" } });
		const fresh = await createCursor({ host: moved.host })(
			"composer-2.5",
		).doStream(call(prompt("run-7#2")));
		expect(moved.opened[0].agentId).toBeUndefined();
		expect(moved.sent[0].request.message.text).toContain(
			"<conversation_history>",
		);
		expect(moved.sent[0].request.message.text).toMatch(/Again$/);
		expect(fresh.request?.body).toMatchObject({ route: "new" });
	});

	it("maps a call's reasoning onto the model's own effort parameter", async () => {
		const fake = fakeHost();
		await createCursor({ host: fake.host })("claude-opus-5-5").doStream(
			call([user("Hi")], { reasoning: "xhigh" }),
		);
		expect(fake.opened[0].options.model).toEqual({
			id: "claude-opus-5-5",
			params: [{ id: "effort", value: "xhigh" }],
		});

		const warnings: any[] = [];
		const gpt = {
			id: "gpt",
			parameters: [
				{
					id: "reasoning",
					values: ["low", "medium", "high", "extra-high"].map((value) => ({
						value,
					})),
				},
			],
		};
		expect(modelParams(gpt, "xhigh", undefined, warnings)).toEqual([
			{ id: "reasoning", value: "extra-high" },
		]);
		const haiku = {
			id: "haiku",
			parameters: [
				{ id: "thinking", values: [{ value: "false" }, { value: "true" }] },
			],
		};
		expect(modelParams(haiku, "none", undefined, warnings)).toEqual([
			{ id: "thinking", value: "false" },
		]);
		expect(warnings).toEqual([]);
		expect(modelParams({ id: "x" }, "high", undefined, warnings)).toEqual([]);
		expect(warnings).toHaveLength(1);
	});

	it("steers a running run with what the user says meanwhile", async () => {
		const fake = fakeHost();
		const provider = createCursor({ host: fake.host });
		const first = await provider("composer-2.5").doStream(
			call([user("Do it")], { tools: [weather] }),
		);
		const toolCall = clientCall("get_weather", {});
		fake.emit(started("c1", toolCall), listed);
		const parts = await read(first.stream);
		const callPart = parts.find((part) => part.type === "tool-call") as any;
		await provider("composer-2.5").doStream(
			call(
				[
					user("Do it"),
					{
						role: "assistant",
						content: [
							{
								type: "tool-call",
								toolCallId: "c1",
								toolName: "get_weather",
								input: {},
								providerOptions: { cursor: callPart.providerMetadata.cursor },
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
								output: { type: "text", value: "ok" },
							},
						],
					},
					user("Also check Rome"),
				],
				{ tools: [weather] },
			),
		);
		expect(fake.steered).toEqual(["Also check Rome"]);
		expect(fake.sent).toHaveLength(1);
	});
});
