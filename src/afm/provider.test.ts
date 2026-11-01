import type { AfmEvent, AfmizeClient, AfmRequest } from "@afmize/node";
import type {
	LanguageModelV4Message,
	LanguageModelV4StreamPart,
} from "@ai-sdk/provider";
import { createAfmProvider } from "#afm/provider.ts";
import { AfmError } from "#afm/stream.ts";
import type { AfmOptions } from "#afm/types.ts";

/** A client that answers every request with `events`, keeping the requests. */
const fake = (events: AfmEvent[]) => {
	const requests: AfmRequest[] = [];
	const client: AfmizeClient = {
		availability: async () => ({
			onDevice: { available: true },
			privateCloudCompute: { available: false, reason: "os-not-supported" },
		}),
		stream: async function* (request) {
			requests.push(request);
			yield* events;
		},
	};
	return { client, requests };
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

describe("afm/provider", () => {
	it("converts the call to afmize's request", async () => {
		const { client, requests } = fake([
			{ type: "stream-start" },
			{ type: "finish", finishReason: "stop" },
		]);
		const { warnings } = await createAfmProvider({ afmize: client })(
			"on-device",
		).doGenerate({
			prompt: [
				{ role: "system", content: "Be brief." },
				{
					role: "user",
					content: [
						{ type: "text", text: "What is this?" },
						{
							type: "file",
							mediaType: "image",
							data: { type: "data", data: new Uint8Array([1, 2, 3]) },
						},
						{
							type: "file",
							mediaType: "application/pdf",
							data: { type: "url", url: new URL("https://x.dev/a.pdf") },
						},
					],
				},
			],
			temperature: 0.5,
			maxOutputTokens: 64,
			topK: 3,
			reasoning: "high",
			tools: [
				{
					type: "function",
					name: "weather",
					inputSchema: { type: "object" },
				},
				{ type: "function", name: "time", inputSchema: { type: "object" } },
			],
			toolChoice: { type: "tool", toolName: "weather" },
		});

		expect(requests).toEqual([
			{
				model: "on-device",
				temperature: 0.5,
				maximumResponseTokens: 64,
				reasoningLevel: "deep",
				messages: [
					{ role: "system", parts: [{ type: "text", text: "Be brief." }] },
					{
						role: "user",
						parts: [
							{ type: "text", text: "What is this?" },
							{ type: "file", mediaType: "image/*", data: "AQID" },
						],
					},
				],
				tools: [
					{
						name: "weather",
						description: undefined,
						inputSchema: { type: "object" },
					},
				],
				toolChoice: "required",
			},
		]);
		expect(warnings).toEqual([
			{ type: "unsupported", feature: "topK" },
			expect.objectContaining({ feature: "application/pdf files" }),
		]);
	});

	it("prefers providerOptions.afm.reasoningLevel", async () => {
		const { client, requests } = fake([
			{ type: "finish", finishReason: "stop" },
		]);
		await createAfmProvider({ afmize: () => client })(
			"private-cloud-compute",
		).doGenerate({
			prompt: [user("Hi")],
			reasoning: "high",
			providerOptions: {
				afm: { reasoningLevel: "light" } satisfies AfmOptions,
			},
		});
		expect(requests[0]?.reasoningLevel).toBe("light");
	});

	it("streams blocks, tool calls and usage", async () => {
		const { client } = fake([
			{ type: "stream-start" },
			{ type: "reasoning-delta", id: "r", delta: "Hmm" },
			{ type: "text-delta", id: "t", delta: "Hel" },
			{ type: "text-replace", id: "t", text: "Hello" },
			{
				type: "tool-call",
				toolCallId: "c",
				toolName: "weather",
				input: '{"city":"Paris"}',
			},
			{
				type: "finish",
				finishReason: "tool-calls",
				usage: {
					inputTokens: 10,
					cachedInputTokens: 4,
					outputTokens: 8,
					reasoningTokens: 3,
					totalTokens: 18,
				},
			},
		]);
		const { stream } = await createAfmProvider({ afmize: client })(
			"on-device",
		).doStream({ prompt: [user("Hi")] });

		expect((await read(stream)).map((part) => part.type)).toEqual([
			"stream-start",
			"reasoning-start",
			"reasoning-delta",
			"reasoning-end",
			"text-start",
			"text-delta",
			"text-delta",
			"text-end",
			"tool-input-start",
			"tool-input-delta",
			"tool-input-end",
			"tool-call",
			"finish",
		]);

		const result = await createAfmProvider({ afmize: client })(
			"on-device",
		).doGenerate({ prompt: [user("Hi")] });
		expect(result.content).toEqual([
			{ type: "reasoning", text: "Hmm" },
			{ type: "text", text: "Hello" },
			{
				type: "tool-call",
				toolCallId: "c",
				toolName: "weather",
				input: '{"city":"Paris"}',
			},
		]);
		expect(result.finishReason).toEqual({
			unified: "tool-calls",
			raw: "tool-calls",
		});
		expect(result.usage.inputTokens).toEqual({
			total: 10,
			noCache: 6,
			cacheRead: 4,
			cacheWrite: undefined,
		});
		expect(result.usage.outputTokens).toEqual({
			total: 8,
			text: 5,
			reasoning: 3,
		});
	});

	it("reports afmize's errors", async () => {
		const { client } = fake([
			{ type: "stream-start" },
			{ type: "error", code: "guardrail-violation", message: "no" },
			{ type: "finish", finishReason: "error" },
		]);
		const model = createAfmProvider({ afmize: client })("on-device");

		const parts = await read(
			(await model.doStream({ prompt: [user("Hi")] })).stream,
		);
		expect(parts.at(-1)).toMatchObject({
			type: "finish",
			finishReason: { unified: "content-filter", raw: "error" },
		});

		await expect(model.doGenerate({ prompt: [user("Hi")] })).rejects.toThrow(
			AfmError,
		);
	});

	it("rejects unknown models", () => {
		expect(() =>
			createAfmProvider({ afmize: fake([]).client })("gpt" as never),
		).toThrow();
	});
});

// Through @afmize/node, the default, against the model itself.
describe.runIf(process.platform === "darwin")("afm/provider (live)", () => {
	const afm = createAfmProvider();

	it("generates a response, where Apple Intelligence is on", async ({
		skip,
	}) => {
		const { onDevice } = await afm.availability();
		if (!onDevice.available) skip(onDevice.reason);

		const result = await afm("on-device").doGenerate({
			prompt: [user("Say hello in three words.")],
			maxOutputTokens: 32,
		});
		expect(result.finishReason.unified).toBe("stop");
		expect(result.content).toEqual([
			expect.objectContaining({ type: "text", text: expect.any(String) }),
		]);
	});
});
