import {
	type CodexChannel,
	type CodexChannelMessage,
	connectCodexHost,
	serveCodexHost,
} from "#codex/host/channel.ts";
import type { CodexHost, CodexServerMessage } from "#codex/host/protocol.ts";

/** Two ends of a channel, delivering in order but never synchronously. */
const pair = (): [CodexChannel, CodexChannel] => {
	const handlers: [
		Set<(message: CodexChannelMessage) => void>,
		Set<(message: CodexChannelMessage) => void>,
	] = [new Set(), new Set()];
	const end = (mine: 0 | 1): CodexChannel => ({
		post: (message) => {
			const copy = JSON.parse(JSON.stringify(message));
			queueMicrotask(() => {
				for (const handler of handlers[mine === 0 ? 1 : 0]) handler(copy);
			});
		},
		listen: (handler) => {
			handlers[mine].add(handler);
			return () => handlers[mine].delete(handler);
		},
	});
	return [end(0), end(1)];
};

describe("codex/host/channel", () => {
	it("relays requests, answers and what the app-server says", async () => {
		const listeners = new Set<(message: CodexServerMessage) => void>();
		const responded: unknown[] = [];
		const host: CodexHost = {
			status: async () => ({ installed: true, loggedIn: true, version: "1" }),
			request: async (method, params) => {
				if (method === "fail") throw new Error("Nope");
				return { method, params };
			},
			respond: async (id, response) => {
				responded.push({ id, response });
			},
			listen: (listener) => {
				listeners.add(listener);
				return () => listeners.delete(listener);
			},
			history: async (threadId) => [{ type: "turn", turnId: threadId }],
		};
		const [served, client] = pair();
		serveCodexHost(host, served);
		const remote = connectCodexHost(client);

		const seen: CodexServerMessage[] = [];
		remote.listen((message) => seen.push(message));

		expect(await remote.status()).toMatchObject({ version: "1" });
		expect(await remote.request("thread/start", { a: 1 })).toEqual({
			method: "thread/start",
			params: { a: 1 },
		});
		await expect(remote.request("fail", {})).rejects.toThrow("Nope");
		await remote.respond(4, { result: { decision: "accept" } });
		expect(responded).toEqual([
			{ id: 4, response: { result: { decision: "accept" } } },
		]);
		expect(await remote.history("t")).toEqual([{ type: "turn", turnId: "t" }]);

		for (const listener of listeners)
			listener({ method: "item/tool/call", id: 9, params: { threadId: "t" } });
		await new Promise((resolve) => setTimeout(resolve, 0));
		expect(seen).toEqual([
			{ method: "item/tool/call", id: 9, params: { threadId: "t" } },
		]);
	});
});
