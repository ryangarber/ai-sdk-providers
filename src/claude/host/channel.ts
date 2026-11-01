import type { ClaudeCodeHost, ClaudeCodeHostEvent } from "./protocol.ts";

/** The methods a channel carries, other than `start`. */
type Method = Exclude<keyof ClaudeCodeHost, "start">;

/** What goes over a channel between a host and the providers using it. */
export type ClaudeCodeChannelMessage =
	| { kind: "call"; id: number; method: Method | "start"; args: unknown[] }
	| { kind: "return"; id: number; value?: unknown; error?: string }
	| { kind: "event"; key: string; event: ClaudeCodeHostEvent };

/**
 * One end of any JSON message channel: a worker's `postMessage`, a
 * webview's RPC, a socket. Messages must arrive in the order posted.
 */
export interface ClaudeCodeChannel {
	post(message: ClaudeCodeChannelMessage): void;
	/** Listens for messages from the other end; returns how to stop. */
	listen(handler: (message: ClaudeCodeChannelMessage) => void): () => void;
}

const METHODS: Method[] = [
	"status",
	"send",
	"answer",
	"update",
	"interrupt",
	"stopTask",
	"close",
	"hasSession",
];

/**
 * Serves `host` over `channel`, for {@link connectClaudeCodeHost} at the other
 * end. Returns how to stop.
 */
export function serveClaudeCodeHost(
	host: ClaudeCodeHost,
	channel: ClaudeCodeChannel,
): () => void {
	return channel.listen(async (message) => {
		if (message.kind !== "call") return;
		const { id, method, args } = message;
		try {
			let value: unknown;
			if (method === "start") {
				const [request, key] = args as [
					Parameters<ClaudeCodeHost["start"]>[0],
					string,
				];
				value = await host.start(request, (event) =>
					channel.post({ kind: "event", key, event }),
				);
			} else {
				value = await (host[method] as (...args: unknown[]) => unknown)(
					...args,
				);
			}
			channel.post({ kind: "return", id, value });
		} catch (error) {
			channel.post({
				kind: "return",
				id,
				error: error instanceof Error ? error.message : String(error),
			});
		}
	});
}

/** A host served over `channel` by {@link serveClaudeCodeHost}. */
export function connectClaudeCodeHost(
	channel: ClaudeCodeChannel,
): ClaudeCodeHost {
	let next = 0;
	const calls = new Map<
		number,
		{ resolve: (value: unknown) => void; reject: (error: Error) => void }
	>();
	const listeners = new Map<string, (event: ClaudeCodeHostEvent) => void>();

	channel.listen((message) => {
		if (message.kind === "return") {
			const call = calls.get(message.id);
			calls.delete(message.id);
			if (message.error !== undefined) call?.reject(new Error(message.error));
			else call?.resolve(message.value);
		} else if (message.kind === "event") {
			listeners.get(message.key)?.(message.event);
			if (message.event.type === "end") listeners.delete(message.key);
		}
	});

	const call = (method: Method | "start", args: unknown[]) =>
		new Promise<unknown>((resolve, reject) => {
			const id = next++;
			calls.set(id, { resolve, reject });
			channel.post({ kind: "call", id, method, args });
		});

	const host = Object.fromEntries(
		METHODS.map((method) => [
			method,
			(...args: unknown[]) => call(method, args),
		]),
	) as Omit<ClaudeCodeHost, "start">;

	return {
		...host,
		async start(request, onEvent) {
			const key = crypto.randomUUID();
			listeners.set(key, onEvent);
			try {
				return (await call("start", [request, key])) as string;
			} catch (error) {
				listeners.delete(key);
				throw error;
			}
		},
	};
}
