import type {
	ClaudeCodeChannel,
	ClaudeCodeChannelMessage,
} from "#claude/host/channel.ts";
import {
	connectClaudeCodeHost,
	serveClaudeCodeHost,
} from "#claude/host/channel.ts";
import type {
	ClaudeCodeHost,
	ClaudeCodeHostEvent,
} from "#claude/host/protocol.ts";

/** Two ends of a channel that sends everything as JSON, as a real one would. */
const pipe = (): [ClaudeCodeChannel, ClaudeCodeChannel] => {
	const handlers: [
		Set<(m: ClaudeCodeChannelMessage) => void>,
		Set<(m: ClaudeCodeChannelMessage) => void>,
	] = [new Set(), new Set()];
	const end = (self: 0 | 1): ClaudeCodeChannel => ({
		post: (message) => {
			const copy = JSON.parse(JSON.stringify(message));
			queueMicrotask(() => {
				for (const handler of handlers[self === 0 ? 1 : 0]) handler(copy);
			});
		},
		listen: (handler) => {
			handlers[self].add(handler);
			return () => handlers[self].delete(handler);
		},
	});
	return [end(0), end(1)];
};

describe("claude/host/channel", () => {
	it("carries calls, their results and a run's events across", async () => {
		const answers: unknown[] = [];
		const host: ClaudeCodeHost = {
			status: async () => ({ installed: true, loggedIn: false }),
			start: async (_request, onEvent) => {
				setTimeout(() => {
					onEvent({ type: "error", message: "boom" });
					onEvent({ type: "end" });
				}, 1);
				return "run-1";
			},
			send: async () => {},
			answer: async (...args) => {
				answers.push(args);
			},
			update: async () => {},
			interrupt: async () => {},
			stopTask: async () => {},
			close: async () => {},
			hasSession: async () => {
				throw new Error("no disk");
			},
		};
		const [server, client] = pipe();
		serveClaudeCodeHost(host, server);
		const remote = connectClaudeCodeHost(client);

		expect(await remote.status()).toEqual({ installed: true, loggedIn: false });
		await expect(remote.hasSession("x")).rejects.toThrow("no disk");

		const events: ClaudeCodeHostEvent[] = [];
		const ended = new Promise<void>((resolve) => {
			void remote
				.start(
					{
						options: {},
						tools: [],
						mcpServerName: "ai-sdk",
						prompt: {
							type: "user",
							message: { role: "user", content: "hi" },
							parent_tool_use_id: null,
						},
					},
					(event) => {
						events.push(event);
						if (event.type === "end") resolve();
					},
				)
				.then((id) => expect(id).toBe("run-1"));
		});
		await ended;
		expect(events).toEqual([
			{ type: "error", message: "boom" },
			{ type: "end" },
		]);

		await remote.answer("run-1", "q", { type: "tool-result", content: [] });
		expect(answers).toEqual([
			["run-1", "q", { type: "tool-result", content: [] }],
		]);
	});
});
