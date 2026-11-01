import type { CodexHost, CodexServerMessage } from "./protocol.ts";

/** The methods a channel carries; `listen` is its event stream. */
type Method = Exclude<keyof CodexHost, "listen">;

/** What goes over a channel between a host and the providers using it. */
export type CodexChannelMessage =
	| { kind: "call"; id: number; method: Method; args: unknown[] }
	| { kind: "return"; id: number; value?: unknown; error?: string }
	| { kind: "event"; message: CodexServerMessage };

/**
 * One end of any JSON message channel: a worker's `postMessage`, a
 * webview's RPC, a socket. Messages must arrive in the order posted.
 */
export interface CodexChannel {
	post(message: CodexChannelMessage): void;
	/** Listens for messages from the other end; returns how to stop. */
	listen(handler: (message: CodexChannelMessage) => void): () => void;
}

const METHODS: Method[] = ["status", "request", "respond", "history"];

/**
 * Serves `host` over `channel`, for {@link connectCodexHost} at the other end.
 * Returns how to stop.
 */
export function serveCodexHost(
	host: CodexHost,
	channel: CodexChannel,
): () => void {
	const stopEvents = host.listen((message) =>
		channel.post({ kind: "event", message }),
	);
	const stopCalls = channel.listen(async (message) => {
		if (message.kind !== "call") return;
		const { id, method, args } = message;
		if (!METHODS.includes(method)) return;
		try {
			const value = await (host[method] as (...args: unknown[]) => unknown)(
				...args,
			);
			channel.post({ kind: "return", id, value });
		} catch (error) {
			channel.post({
				kind: "return",
				id,
				error: error instanceof Error ? error.message : String(error),
			});
		}
	});
	return () => {
		stopEvents();
		stopCalls();
	};
}

/** A host served over `channel` by {@link serveCodexHost}. */
export function connectCodexHost(channel: CodexChannel): CodexHost {
	let next = 0;
	const calls = new Map<
		number,
		{ resolve: (value: unknown) => void; reject: (error: Error) => void }
	>();
	const listeners = new Set<(message: CodexServerMessage) => void>();

	channel.listen((message) => {
		if (message.kind === "return") {
			const call = calls.get(message.id);
			calls.delete(message.id);
			if (message.error !== undefined) call?.reject(new Error(message.error));
			else call?.resolve(message.value);
		} else if (message.kind === "event")
			for (const listener of listeners) listener(message.message);
	});

	const call = (method: Method, args: unknown[]) =>
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
	) as Omit<CodexHost, "listen">;

	return {
		...host,
		listen(handler) {
			listeners.add(handler);
			return () => listeners.delete(handler);
		},
	};
}
