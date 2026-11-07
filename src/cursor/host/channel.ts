import type { CursorHost, CursorHostEvent } from "./protocol.ts";

/** The methods a channel carries, other than `send`. */
type Method = Exclude<keyof CursorHost, "send">;

/** What goes over a channel between a host and the providers using it. */
export type CursorChannelMessage =
	| { kind: "call"; id: number; method: Method | "send"; args: unknown[] }
	| { kind: "return"; id: number; value?: unknown; error?: string }
	| { kind: "event"; key: string; event: CursorHostEvent };

/**
 * One end of any JSON message channel: a worker's `postMessage`, a
 * webview's RPC, a socket. Messages must arrive in the order posted.
 */
export interface CursorChannel {
	post(message: CursorChannelMessage): void;
	/** Listens for messages from the other end; returns how to stop. */
	listen(handler: (message: CursorChannelMessage) => void): () => void;
}

const METHODS: Method[] = [
	"status",
	"open",
	"answer",
	"steer",
	"cancel",
	"close",
	"lastRun",
];

/**
 * Serves `host` over `channel`, for {@link connectCursorHost} at the other
 * end. Returns how to stop.
 */
export function serveCursorHost(
	host: CursorHost,
	channel: CursorChannel,
): () => void {
	return channel.listen(async (message) => {
		if (message.kind !== "call") return;
		const { id, method, args } = message;
		if (method !== "send" && !METHODS.includes(method)) return;
		try {
			let value: unknown;
			if (method === "send") {
				const [agentId, request, key] = args as [
					string,
					Parameters<CursorHost["send"]>[1],
					string,
				];
				value = await host.send(agentId, request, (event) =>
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

/** A host served over `channel` by {@link serveCursorHost}. */
export function connectCursorHost(channel: CursorChannel): CursorHost {
	let next = 0;
	const calls = new Map<
		number,
		{ resolve: (value: unknown) => void; reject: (error: Error) => void }
	>();
	const listeners = new Map<string, (event: CursorHostEvent) => void>();

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

	const call = (method: Method | "send", args: unknown[]) =>
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
	) as Omit<CursorHost, "send">;

	return {
		...host,
		async send(agentId, request, onEvent) {
			const key = crypto.randomUUID();
			listeners.set(key, onEvent);
			try {
				return (await call("send", [agentId, request, key])) as string;
			} catch (error) {
				listeners.delete(key);
				throw error;
			}
		},
	};
}
