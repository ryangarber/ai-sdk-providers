import type {
	LanguageModelV4CallOptions,
	LanguageModelV4Message,
	LanguageModelV4Prompt,
	LanguageModelV4StreamPart,
} from "@ai-sdk/provider";
import { rebuildItems, sliceHistory } from "#codex/history.ts";
import type { CodexResponseItem } from "#codex/host/app-server.ts";
import type {
	CodexHistoryEntry,
	CodexHost,
	CodexServerMessage,
} from "#codex/host/protocol.ts";
import { createCodex } from "#codex/provider.ts";

interface Call {
	method: string;
	params: any;
}

/** A host that records what it is asked and emits what a test says. */
const fakeHost = (saved: Record<string, CodexHistoryEntry[]> = {}) => {
	const calls: Call[] = [];
	const responses: { id: unknown; response: any }[] = [];
	let handler: ((message: CodexServerMessage) => void) | undefined;
	let threads = 0;
	let turns = 0;
	const host: CodexHost = {
		status: async () => ({ installed: true, loggedIn: true }),
		async request(method, params) {
			calls.push({ method, params });
			switch (method) {
				case "thread/start":
					return { thread: { id: `thread-${++threads}` } };
				case "turn/start":
					return { turn: { id: `turn-${++turns}`, status: "inProgress" } };
				case "turn/steer":
					return { turnId: (params as any).expectedTurnId };
				case "config/read":
					return { config: { mcp_servers: { zeta: {}, alpha: {} } } };
				default:
					return {};
			}
		},
		respond: async (id, response) => {
			responses.push({ id, response });
		},
		listen: (listener) => {
			handler = listener;
			return () => {};
		},
		history: async (threadId) => saved[threadId],
	};
	const emit = (...messages: CodexServerMessage[]) => {
		for (const message of messages) handler?.(message);
	};
	const sent = (method: string) =>
		calls.filter((call) => call.method === method);
	return { host, calls, responses, emit, sent };
};

/** Messages of a thread's turn, as app-server sends them. */
const on = (threadId: string, turnId: string) => ({
	turnStarted: (): CodexServerMessage => ({
		method: "turn/started",
		params: { threadId, turn: { id: turnId, status: "inProgress", items: [] } },
	}),
	raw: (item: CodexResponseItem): CodexServerMessage => ({
		method: "rawResponseItem/completed",
		params: { threadId, turnId, item },
	}),
	responseDone: (responseId: string): CodexServerMessage => ({
		method: "rawResponse/completed",
		params: {
			threadId,
			turnId,
			responseId,
			usage: {
				totalTokens: 110,
				inputTokens: 100,
				cachedInputTokens: 60,
				cacheWriteInputTokens: 0,
				outputTokens: 10,
				reasoningOutputTokens: 4,
			},
		},
	}),
	started: (item: any): CodexServerMessage => ({
		method: "item/started",
		params: { threadId, turnId, item },
	}),
	completed: (item: any): CodexServerMessage => ({
		method: "item/completed",
		params: { threadId, turnId, item },
	}),
	delta: (itemId: string, delta: string): CodexServerMessage => ({
		method: "item/agentMessage/delta",
		params: { threadId, turnId, itemId, delta },
	}),
	turnCompleted: (status = "completed"): CodexServerMessage => ({
		method: "turn/completed",
		params: { threadId, turn: { id: turnId, status, items: [], error: null } },
	}),
	request: (id: number, method: string, params: any): CodexServerMessage => ({
		id,
		method,
		params: { threadId, turnId, ...params },
	}),
});

/** A whole reply: a message the model streams, and the end of its response. */
const reply = (
	t: ReturnType<typeof on>,
	id: string,
	text: string,
	response: string,
) => [
	t.started({ type: "agentMessage", id, text: "", phase: "final_answer" }),
	t.delta(id, text),
	t.completed({ type: "agentMessage", id, text, phase: "final_answer" }),
	t.raw({
		type: "message",
		role: "assistant",
		content: [{ type: "output_text", text }],
	}),
	t.responseDone(response),
];

/** A response that runs a script calling tools, as code mode does. */
const script = (t: ReturnType<typeof on>, callId: string, response: string) => [
	t.raw({
		type: "custom_tool_call",
		call_id: callId,
		name: "exec",
		input: "await tools.get_weather({city: 'Paris'})",
	}),
	t.responseDone(response),
];

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

const weather = {
	type: "function" as const,
	name: "get_weather",
	description: "The weather",
	inputSchema: { type: "object", properties: { city: { type: "string" } } },
};

const options = (
	prompt: LanguageModelV4Prompt,
	extra: Partial<LanguageModelV4CallOptions> = {},
): LanguageModelV4CallOptions => ({ prompt, ...extra });

const finishOf = (parts: LanguageModelV4StreamPart[]) =>
	parts.find((part) => part.type === "finish") as Extract<
		LanguageModelV4StreamPart,
		{ type: "finish" }
	>;
const codexMeta = (parts: LanguageModelV4StreamPart[]) =>
	finishOf(parts).providerMetadata?.codex as Record<string, any>;

describe("codex/provider", () => {
	it("starts a thread with the conversation so far, and streams its turn", async () => {
		const fake = fakeHost();
		const model = createCodex({ host: fake.host })("gpt-test");
		const { stream } = await model.doStream(
			options([
				{ role: "system", content: "Be terse." },
				user("Hi"),
				{ role: "assistant", content: [{ type: "text", text: "Hello!" }] },
				user("What's 2+2?"),
			]),
		);
		const t = on("thread-1", "turn-1");
		fake.emit(
			t.turnStarted(),
			...reply(t, "msg-1", "4", "resp-1"),
			t.turnCompleted(),
		);
		const parts = await read(stream);

		const [start] = fake.sent("thread/start");
		expect(start.params).toMatchObject({
			model: "gpt-test",
			baseInstructions: "Be terse.",
			experimentalRawEvents: true,
			dynamicTools: [],
			sandbox: "read-only",
		});
		expect(start.params.config).toMatchObject({
			"features.shell_tool": false,
			"features.multi_agent": false,
			"tools.experimental_request_user_input.enabled": false,
			web_search: "disabled",
			project_doc_max_bytes: 0,
			mcp_servers: { alpha: { enabled: false }, zeta: { enabled: false } },
		});
		const [turn] = fake.sent("turn/start");
		expect(turn.params.input[0].text).toContain("## User\n\nHi");
		expect(turn.params.input[0].text).toContain("## Assistant\n\nHello!");
		expect(turn.params.input[1].text).toBe("What's 2+2?");

		expect(parts.filter((p) => p.type === "text-delta")).toEqual([
			{ type: "text-delta", id: "msg-1", delta: "4" },
		]);
		const finish = finishOf(parts);
		expect(finish.finishReason).toEqual({ unified: "stop", raw: "completed" });
		expect(finish.usage.inputTokens).toMatchObject({
			total: 100,
			noCache: 40,
			cacheRead: 60,
		});
		expect(finish.usage.outputTokens).toMatchObject({ text: 6, reasoning: 4 });
		expect(codexMeta(parts)).toMatchObject({
			threadId: "thread-1",
			turnId: "turn-1",
			cursor: "resp-1",
			turnEnded: true,
			route: "new",
		});
	});

	it("takes the instructions preset and built-in tools separately", async () => {
		const fake = fakeHost();
		const codex = createCodex({ host: fake.host });
		await codex("gpt-test").doStream(
			options([{ role: "system", content: "Be terse." }, user("Hi")], {
				tools: [
					weather,
					{
						type: "provider",
						id: "codex.apply_patch" as const,
						name: "patch",
						args: {},
					},
					{
						type: "provider",
						id: "codex.request_user_input" as const,
						name: "ask",
						args: {},
					},
				],
				providerOptions: {
					codex: {
						systemPromptPreset: "codex",
						builtinTools: ["exec_command"],
						userConfig: true,
					},
				},
			}),
		);
		const [start] = fake.sent("thread/start");
		expect(start.params.developerInstructions).toBe("Be terse.");
		expect(start.params.baseInstructions).toBeUndefined();
		expect(start.params.sandbox).toBeUndefined();
		expect(start.params.dynamicTools).toEqual([{ ...weather }]);
		expect(start.params.config).toMatchObject({
			"features.shell_tool": true,
			"features.view_image": false,
			"tools.experimental_request_user_input.enabled": true,
		});
		expect(start.params.config.mcp_servers).toBeUndefined();
		expect(fake.sent("config/read")).toHaveLength(0);
	});

	it("ends a step on the caller's tool calls, and goes on with their results", async () => {
		const fake = fakeHost();
		const model = createCodex({ host: fake.host })("gpt-test");
		const tools = [weather];
		const prompt: LanguageModelV4Prompt = [user("Weather in Paris?")];
		const first = await model.doStream(options(prompt, { tools }));
		const t = on("thread-1", "turn-1");
		fake.emit(
			t.turnStarted(),
			...script(t, "call-exec", "resp-1"),
			t.started({
				type: "dynamicToolCall",
				id: "exec-1",
				tool: "get_weather",
				arguments: { city: "Paris" },
				status: "inProgress",
			}),
			t.request(7, "item/tool/call", {
				callId: "exec-1",
				tool: "get_weather",
				arguments: { city: "Paris" },
			}),
		);
		const parts = await read(first.stream);
		expect(parts.filter((p) => p.type === "tool-call")).toMatchObject([
			{
				toolCallId: "exec-1",
				toolName: "get_weather",
				input: '{"city":"Paris"}',
			},
		]);
		expect(
			(parts.find((p) => p.type === "tool-call") as any).providerExecuted,
		).toBeUndefined();
		expect(finishOf(parts).finishReason.unified).toBe("tool-calls");
		expect(codexMeta(parts).awaiting).toEqual(["tools"]);

		prompt.push(toMessage(parts), {
			role: "tool",
			content: [
				{
					type: "tool-result",
					toolCallId: "exec-1",
					toolName: "get_weather",
					output: { type: "text", value: "Rainy" },
				},
			],
		});
		const second = await model.doStream(options(prompt, { tools }));
		expect(fake.responses).toEqual([
			{
				id: 7,
				response: {
					result: {
						contentItems: [{ type: "inputText", text: "Rainy" }],
						success: true,
					},
				},
			},
		]);
		fake.emit(
			t.completed({
				type: "dynamicToolCall",
				id: "exec-1",
				tool: "get_weather",
				arguments: { city: "Paris" },
				status: "completed",
				contentItems: [{ type: "inputText", text: "Rainy" }],
				success: true,
			}),
			t.raw({
				type: "custom_tool_call_output",
				call_id: "call-exec",
				output: "Rainy",
			}),
			...reply(t, "msg-2", "It's rainy.", "resp-2"),
			t.turnCompleted(),
		);
		const next = await read(second.stream);
		expect(codexMeta(next)).toMatchObject({
			route: "continue",
			cursor: "resp-2",
		});
		expect(next.filter((p) => p.type === "tool-result")).toHaveLength(0);
		expect(next.find((p) => p.type === "text-delta")).toMatchObject({
			delta: "It's rainy.",
		});
		expect(finishOf(next).finishReason.unified).toBe("stop");
		expect(fake.sent("thread/start")).toHaveLength(1);
	});

	it("ends a step once its calls ran, and streams the next response in the next", async () => {
		const fake = fakeHost();
		const model = createCodex({ host: fake.host })("gpt-test", {
			builtinTools: ["exec_command"],
		});
		const prompt: LanguageModelV4Prompt = [user("List files")];
		const first = await model.doStream(options(prompt));
		const t = on("thread-1", "turn-1");
		const command = {
			type: "commandExecution",
			id: "exec-2",
			command: "ls",
			cwd: "/w",
			commandActions: [],
			status: "inProgress",
			aggregatedOutput: null,
			exitCode: null,
		};
		fake.emit(
			t.turnStarted(),
			...script(t, "call-exec", "resp-1"),
			t.started(command),
			t.completed({
				...command,
				status: "completed",
				aggregatedOutput: "a\nb",
				exitCode: 0,
			}),
			t.raw({
				type: "custom_tool_call_output",
				call_id: "call-exec",
				output: "a\nb",
			}),
			...reply(t, "msg-2", "Two files.", "resp-2"),
			t.turnCompleted(),
		);
		const parts = await read(first.stream);
		expect(parts.filter((p) => p.type === "tool-result")).toMatchObject([
			{
				toolCallId: "exec-2",
				toolName: "exec_command",
				result: { output: "a\nb", exitCode: 0, status: "completed" },
			},
		]);
		expect(parts.some((p) => p.type === "text-delta")).toBe(false);
		expect(finishOf(parts).finishReason.unified).toBe("tool-calls");
		expect(codexMeta(parts)).toMatchObject({
			awaiting: ["continue"],
			cursor: "resp-1",
		});

		prompt.push(toMessage(parts));
		const second = await read((await model.doStream(options(prompt))).stream);
		expect(second.find((p) => p.type === "text-delta")).toMatchObject({
			delta: "Two files.",
		});
		expect(codexMeta(second)).toMatchObject({
			route: "continue",
			turnEnded: true,
		});
	});

	it("asks for approval in the same step as the call, again if it must", async () => {
		const fake = fakeHost();
		const tools = [
			weather,
			{
				type: "provider" as const,
				id: "codex.exec_command" as const,
				name: "shell",
				args: {},
			},
		];
		const model = createCodex({ host: fake.host })("gpt-test");
		const prompt: LanguageModelV4Prompt = [user("Do both")];
		const first = await model.doStream(options(prompt, { tools }));
		const t = on("thread-1", "turn-1");
		const command = {
			type: "commandExecution",
			id: "exec-3",
			command: "rm x",
			cwd: "/w",
			commandActions: [],
			status: "inProgress",
		};
		fake.emit(
			t.turnStarted(),
			...script(t, "call-exec", "resp-1"),
			t.started({
				type: "dynamicToolCall",
				id: "exec-1",
				tool: "get_weather",
				arguments: {},
			}),
			t.request(1, "item/tool/call", {
				callId: "exec-1",
				tool: "get_weather",
				arguments: {},
			}),
		);
		const parts = await read(first.stream);
		expect(codexMeta(parts).awaiting).toEqual(["tools"]);

		// The command only starts once the caller's tool returns.
		prompt.push(toMessage(parts), {
			role: "tool",
			content: [
				{
					type: "tool-result",
					toolCallId: "exec-1",
					toolName: "get_weather",
					output: { type: "text", value: "Sunny" },
				},
			],
		});
		const second = await model.doStream(options(prompt, { tools }));
		fake.emit(
			t.started(command),
			t.request(2, "item/commandExecution/requestApproval", {
				itemId: "exec-3",
				command: "rm x",
				reason: "Writes outside the sandbox",
			}),
		);
		const asked = await read(second.stream);
		const call = asked.find((p) => p.type === "tool-call");
		const approval = asked.find((p) => p.type === "tool-approval-request");
		expect(call).toMatchObject({
			toolCallId: "exec-3",
			toolName: "shell",
			providerExecuted: true,
		});
		expect(approval).toMatchObject({
			approvalId: "exec-3",
			toolCallId: "exec-3",
			providerMetadata: { codex: { reason: "Writes outside the sandbox" } },
		});
		expect(asked.indexOf(call as any)).toBeLessThan(
			asked.indexOf(approval as any),
		);
		expect(codexMeta(asked).awaiting).toEqual(["approval"]);

		prompt.push(toMessage(asked), {
			role: "tool",
			content: [
				{
					type: "tool-approval-response",
					approvalId: "exec-3",
					approved: false,
					reason: "Not that file",
				},
			],
		});
		await model.doStream(options(prompt, { tools }));
		expect(fake.responses.at(-1)).toEqual({
			id: 2,
			response: { result: { decision: "decline" } },
		});
		// The reason reaches the model, as Codex takes no message with a denial.
		expect(fake.sent("turn/steer").at(-1)?.params.input[0].text).toBe(
			"The user denied the exec_command call: Not that file",
		);
	});

	it("routes request_user_input to the caller and answers by question", async () => {
		const fake = fakeHost();
		const tools = [
			{
				type: "provider" as const,
				id: "codex.request_user_input" as const,
				name: "ask",
				args: {},
			},
		];
		const model = createCodex({ host: fake.host })("gpt-test");
		const prompt: LanguageModelV4Prompt = [user("Ask me")];
		const first = await model.doStream(options(prompt, { tools }));
		const t = on("thread-1", "turn-1");
		const questions = [
			{
				id: "drink",
				header: "Drink",
				question: "Tea or coffee?",
				isOther: true,
				isSecret: false,
				options: [{ label: "Tea", description: "" }],
			},
		];
		fake.emit(
			t.turnStarted(),
			t.raw({
				type: "function_call",
				call_id: "call-ask",
				name: "request_user_input",
				arguments: "{}",
			}),
			t.request(3, "item/tool/requestUserInput", {
				itemId: "call-ask",
				questions,
			}),
			t.responseDone("resp-1"),
		);
		const parts = await read(first.stream);
		expect(parts.find((p) => p.type === "tool-call")).toMatchObject({
			toolCallId: "call-ask",
			toolName: "ask",
			input: JSON.stringify({ questions }),
		});
		expect(codexMeta(parts).awaiting).toEqual(["tools"]);

		prompt.push(toMessage(parts), {
			role: "tool",
			content: [
				{
					type: "tool-result",
					toolCallId: "call-ask",
					toolName: "ask",
					output: {
						type: "text",
						value: '```json\n{"answers": {"Tea or coffee?": "Tea"}}\n```',
					},
				},
			],
		});
		await model.doStream(options(prompt, { tools }));
		expect(fake.responses).toEqual([
			{
				id: 3,
				response: { result: { answers: { drink: { answers: ["Tea"] } } } },
			},
		]);
	});

	it("sends what the user said meanwhile into the running turn", async () => {
		const fake = fakeHost();
		const model = createCodex({ host: fake.host })("gpt-test");
		const prompt: LanguageModelV4Prompt = [user("Go")];
		const first = await model.doStream(options(prompt, { tools: [weather] }));
		const t = on("thread-1", "turn-1");
		fake.emit(
			t.turnStarted(),
			...script(t, "call-exec", "resp-1"),
			t.started({
				type: "dynamicToolCall",
				id: "exec-1",
				tool: "get_weather",
				arguments: {},
			}),
			t.request(1, "item/tool/call", {
				callId: "exec-1",
				tool: "get_weather",
				arguments: {},
			}),
		);
		const parts = await read(first.stream);
		prompt.push(
			toMessage(parts),
			{
				role: "tool",
				content: [
					{
						type: "tool-result",
						toolCallId: "exec-1",
						toolName: "get_weather",
						output: { type: "text", value: "Rainy" },
					},
				],
			},
			user("Also check Rome"),
		);
		await model.doStream(options(prompt, { tools: [weather] }));
		expect(fake.sent("turn/steer")).toMatchObject([
			{
				params: {
					threadId: "thread-1",
					expectedTurnId: "turn-1",
					input: [{ type: "text", text: "Also check Rome" }],
				},
			},
		]);
	});

	it("starts the next turn in the same thread after one ended", async () => {
		const fake = fakeHost();
		const model = createCodex({ host: fake.host })("gpt-test");
		const prompt: LanguageModelV4Prompt = [user("Hi")];
		const first = await model.doStream(options(prompt));
		const t = on("thread-1", "turn-1");
		fake.emit(
			t.turnStarted(),
			...reply(t, "msg-1", "Hello", "resp-1"),
			t.turnCompleted(),
		);
		prompt.push(toMessage(await read(first.stream)), user("Again"));
		await model.doStream(options(prompt, { reasoning: "high" }));
		expect(fake.sent("thread/start")).toHaveLength(1);
		expect(fake.sent("turn/start").at(-1)?.params).toMatchObject({
			threadId: "thread-1",
			effort: "high",
			input: [{ type: "text", text: "Again" }],
		});
	});

	it("rebuilds a saved thread at the cursor, answering what it left waiting", async () => {
		const t = on("thread-9", "turn-9");
		const saved: CodexHistoryEntry[] = [
			{ type: "turn", turnId: "turn-9" },
			{
				type: "item",
				turnId: "turn-9",
				item: {
					type: "message",
					role: "developer",
					content: [{ type: "input_text", text: "You are Codex" }],
				},
			},
			{
				type: "item",
				turnId: "turn-9",
				item: {
					type: "message",
					role: "user",
					content: [
						{
							type: "input_text",
							text: "<environment_context>\n<cwd>/w</cwd>\n</environment_context>",
						},
					],
				},
			},
			{
				type: "item",
				turnId: "turn-9",
				item: {
					type: "message",
					role: "user",
					content: [{ type: "input_text", text: "Weather?" }],
				},
			},
			{
				type: "item",
				turnId: "turn-9",
				item: {
					type: "custom_tool_call",
					call_id: "call-exec",
					name: "exec",
					input: "…",
					internal_chat_message_metadata_passthrough: {},
				},
			},
			{ type: "response", turnId: "turn-9", responseId: "resp-9" },
		];
		const fake = fakeHost({ "thread-9": saved });
		const model = createCodex({ host: fake.host })("gpt-test");
		const meta = {
			codex: {
				threadId: "thread-9",
				turnId: "turn-9",
				cursor: "resp-9",
				fingerprint: "other",
			},
		};
		const prompt: LanguageModelV4Prompt = [
			user("Weather?"),
			{
				role: "assistant",
				content: [
					{
						type: "tool-call",
						toolCallId: "exec-1",
						toolName: "get_weather",
						input: { city: "Paris" },
						providerOptions: meta,
					},
					{ type: "custom", kind: "codex.step", providerOptions: meta },
				],
			},
			{
				role: "tool",
				content: [
					{
						type: "tool-result",
						toolCallId: "exec-1",
						toolName: "get_weather",
						output: { type: "text", value: "Rainy" },
					},
				],
			},
		];
		const { stream } = await model.doStream(
			options(prompt, { tools: [weather] }),
		);
		const [inject] = fake.sent("thread/inject_items");
		expect(inject.params).toEqual({
			threadId: "thread-1",
			items: [
				{
					type: "message",
					role: "user",
					content: [{ type: "input_text", text: "Weather?" }],
				},
				{
					type: "custom_tool_call",
					call_id: "call-exec",
					name: "exec",
					input: "…",
				},
				{
					type: "custom_tool_call_output",
					call_id: "call-exec",
					output:
						"The script stopped when the session restarted, so it didn't finish. What its calls got:\nget_weather: Rainy",
				},
			],
		});
		expect(fake.sent("turn/start")[0].params.input).toEqual([
			{ type: "text", text: "Continue.", text_elements: [] },
		]);
		const r = on("thread-1", "turn-1");
		fake.emit(
			r.turnStarted(),
			...reply(r, "msg-1", "Rainy.", "resp-1"),
			r.turnCompleted(),
		);
		const parts = await read(stream);
		expect(codexMeta(parts)).toMatchObject({
			route: "rebuild",
			threadId: "thread-1",
		});
		void t;
	});

	it("rebuilds the live thread when the call's tools change, and closes it", async () => {
		const fake = fakeHost();
		const model = createCodex({ host: fake.host })("gpt-test");
		const prompt: LanguageModelV4Prompt = [user("Hi")];
		const first = await model.doStream(options(prompt, { tools: [weather] }));
		const t = on("thread-1", "turn-1");
		fake.emit(
			t.turnStarted(),
			t.raw({
				type: "message",
				role: "user",
				content: [{ type: "input_text", text: "Hi" }],
			}),
			...reply(t, "msg-1", "Hello", "resp-1"),
			t.turnCompleted(),
		);
		prompt.push(toMessage(await read(first.stream)), user("Now the time"));
		const time = { ...weather, name: "get_time" };
		await model.doStream(options(prompt, { tools: [weather, time] }));
		expect(fake.sent("thread/start")).toHaveLength(2);
		expect(fake.sent("thread/start")[1].params.dynamicTools).toHaveLength(2);
		expect(fake.sent("thread/inject_items")[0].params.items).toEqual([
			{
				type: "message",
				role: "user",
				content: [{ type: "input_text", text: "Hi" }],
			},
			{
				type: "message",
				role: "assistant",
				content: [{ type: "output_text", text: "Hello" }],
			},
		]);
		expect(fake.sent("thread/unsubscribe")).toMatchObject([
			{ params: { threadId: "thread-1" } },
		]);
	});

	it("starts over with a transcript when the thread can't be read", async () => {
		const fake = fakeHost();
		const model = createCodex({ host: fake.host })("gpt-test");
		const meta = {
			codex: { threadId: "gone", turnId: "turn-x", cursor: "resp-x" },
		};
		await model.doStream(
			options([
				user("Hi"),
				{
					role: "assistant",
					content: [{ type: "text", text: "Hello", providerOptions: meta }],
				},
				user("Still there?"),
			]),
		);
		expect(fake.sent("thread/inject_items")).toHaveLength(0);
		const input = fake.sent("turn/start")[0].params.input;
		expect(input[0].text).toContain("<conversation_history>");
		expect(input[1].text).toBe("Still there?");
	});

	it("ignores items given to a thread, which come back before its turn", async () => {
		const fake = fakeHost();
		const model = createCodex({ host: fake.host })("gpt-test");
		const { stream } = await model.doStream(options([user("Hi")]));
		const t = on("thread-1", "turn-1");
		fake.emit(
			{
				method: "rawResponseItem/completed",
				params: {
					threadId: "thread-1",
					turnId: "",
					item: {
						type: "custom_tool_call",
						call_id: "old",
						name: "exec",
						input: "",
					},
				},
			},
			t.turnStarted(),
			...reply(t, "msg-1", "Hello", "resp-1"),
			t.turnCompleted(),
		);
		const parts = await read(stream);
		expect(finishOf(parts).finishReason.unified).toBe("stop");
	});

	it("streams a subagent's work as its spawn call's preliminary results", async () => {
		const fake = fakeHost();
		const model = createCodex({ host: fake.host })("gpt-test", {
			builtinTools: ["spawn_agent"],
		});
		const prompt: LanguageModelV4Prompt = [user("Delegate")];
		const { stream } = await model.doStream(options(prompt));
		const t = on("thread-1", "turn-1");
		const c = on("child-1", "child-turn");
		fake.emit(
			t.turnStarted(),
			t.raw({
				type: "function_call",
				call_id: "call-spawn",
				name: "spawn_agent",
				namespace: "collaboration",
				arguments: '{"message":"echo hi"}',
			}),
			t.responseDone("resp-1"),
			t.started({
				type: "subAgentActivity",
				id: "call-spawn",
				kind: "started",
				agentThreadId: "child-1",
				agentPath: "/root/echo",
			}),
			{
				method: "thread/started",
				params: { thread: { id: "child-1", parentThreadId: "thread-1" } },
			},
			c.completed({
				type: "agentMessage",
				id: "c-msg",
				text: "hi",
				phase: "final_answer",
			}),
			t.raw({
				type: "function_call_output",
				call_id: "call-spawn",
				output: "spawned",
			}),
			t.completed({
				type: "subAgentActivity",
				id: "sub-done",
				kind: "completed",
				agentThreadId: "child-1",
				agentPath: "/root/echo",
			}),
			...reply(t, "msg-2", "It said hi.", "resp-2"),
			t.turnCompleted(),
		);
		const parts = await read(stream);
		expect(parts.find((p) => p.type === "tool-call")).toMatchObject({
			toolCallId: "call-spawn",
			toolName: "spawn_agent",
			input: JSON.stringify({
				message: "echo hi",
				agentPath: "/root/echo",
				agentThreadId: "child-1",
			}),
		});
		const results = parts.filter((p) => p.type === "tool-result");
		expect(results).toMatchObject([
			{
				preliminary: true,
				result: {
					type: "subagent",
					status: "running",
					messages: [
						{ role: "assistant", content: [{ type: "text", text: "hi" }] },
					],
				},
			},
			{ toolCallId: "call-spawn", result: "spawned" },
		]);
		// Its call ran: the rest is the next step's, the subagent's end too.
		expect(codexMeta(parts).awaiting).toEqual(["continue"]);
		prompt.push(toMessage(parts));
		const next = await read((await model.doStream(options(prompt))).stream);
		expect(next.filter((p) => p.type === "tool-result")).toMatchObject([
			{
				toolCallId: "call-spawn",
				preliminary: true,
				result: { status: "completed" },
			},
		]);
		expect(finishOf(next).finishReason.unified).toBe("stop");
	});

	it("interrupts the turn when the call is aborted", async () => {
		const fake = fakeHost();
		const model = createCodex({ host: fake.host })("gpt-test");
		const abort = new AbortController();
		const { stream } = await model.doStream(
			options([user("Write a story")], { abortSignal: abort.signal }),
		);
		const t = on("thread-1", "turn-1");
		fake.emit(t.turnStarted(), t.delta("msg-1", "Once"));
		setTimeout(() => abort.abort(), 10);
		const parts = await read(stream);
		expect(finishOf(parts).finishReason.unified).toBe("other");
		expect(parts.some((p) => p.type === "text-end")).toBe(true);
		await new Promise((resolve) => setTimeout(resolve, 0));
		expect(fake.sent("turn/interrupt")).toMatchObject([
			{ params: { threadId: "thread-1", turnId: "turn-1" } },
		]);
	});

	it("aggregates a step in doGenerate", async () => {
		const fake = fakeHost();
		const model = createCodex({ host: fake.host })("gpt-test");
		const done = model.doGenerate(options([user("Hi")]));
		await new Promise((resolve) => setTimeout(resolve, 0));
		const t = on("thread-1", "turn-1");
		fake.emit(
			t.turnStarted(),
			...reply(t, "msg-1", "Hello", "resp-1"),
			t.turnCompleted(),
		);
		const result = await done;
		expect(result.content).toMatchObject([
			{ type: "text", text: "Hello" },
			{ type: "custom", kind: "codex.step" },
		]);
		expect(result.finishReason.unified).toBe("stop");
		expect(result.response?.id).toBe("resp-1");
	});
});

describe("codex/history", () => {
	const item = (text: string, role = "user"): CodexResponseItem => ({
		type: "message",
		role,
		content: [{ type: role === "user" ? "input_text" : "output_text", text }],
	});

	it("slices up to a response and its step's outputs, after compaction", () => {
		const entries: CodexHistoryEntry[] = [
			{ type: "turn", turnId: "a" },
			{ type: "item", turnId: "a", item: item("old") },
			{ type: "compacted", turnId: "a", items: [item("summary")] },
			{ type: "turn", turnId: "b" },
			{ type: "item", turnId: "b", item: item("go") },
			{
				type: "item",
				turnId: "b",
				item: { type: "function_call", call_id: "c1", name: "x" },
			},
			{ type: "response", turnId: "b", responseId: "r1" },
			{
				type: "item",
				turnId: "b",
				item: { type: "function_call_output", call_id: "c1", output: "ok" },
			},
			{ type: "item", turnId: "b", item: item("later", "assistant") },
			{ type: "response", turnId: "b", responseId: "r2" },
		];
		expect(sliceHistory(entries, { turnId: "b", cursor: "r1" })).toEqual([
			item("summary"),
			item("go"),
			{ type: "function_call", call_id: "c1", name: "x" },
			{ type: "function_call_output", call_id: "c1", output: "ok" },
		]);
		expect(
			sliceHistory(entries, { turnId: "b", cursor: "nope" }),
		).toBeUndefined();
	});

	it("answers a question the thread left waiting with the caller's answers", () => {
		const items = rebuildItems(
			[{ type: "function_call", call_id: "q1", name: "request_user_input" }],
			[
				{
					call: { id: "q1", name: "ask", input: {}, providerExecuted: false },
					result: { type: "json", value: { answers: { drink: "Tea" } } },
				},
			],
			{ keepReasoning: true },
		);
		expect(items.at(-1)).toEqual({
			type: "function_call_output",
			call_id: "q1",
			output: '{"answers":{"drink":"Tea"}}',
		});
	});
});
