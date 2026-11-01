import type { CodexReasoningEffort, CodexResponseItem } from "./app-server.ts";

/** The id of a JSON-RPC request either side made. */
export type CodexRequestId = string | number;

/**
 * How a provider reaches Codex: its app-server (`codex app-server`), relayed
 * as JSON so a host can sit across any boundary: in-process
 * ({@link createLocalHost}), or in another process or thread over a message
 * channel (`serveCodexHost` / `connectCodexHost`). The host is thin. It runs
 * the app-server and relays what it says and asks; the provider does the rest.
 */
export interface CodexHost {
	/** Whether `codex` is installed and signed in, and what it offers. */
	status(options?: CodexStatusOptions): Promise<CodexStatus>;
	/** Makes an app-server request (`thread/start`, `turn/start`, …). */
	request(method: string, params: unknown): Promise<unknown>;
	/** Answers a request the app-server made, by its id. */
	respond(id: CodexRequestId, response: CodexResponse): Promise<void>;
	/**
	 * Listens to what the app-server sends: notifications, and requests of its
	 * own (with an `id`) to {@link respond} to. Returns how to stop.
	 */
	listen(handler: (message: CodexServerMessage) => void): () => void;
	/**
	 * A thread's history as it was saved, model-visible items in order, for
	 * rebuilding it at any step. Undefined if it can't be read.
	 */
	history(threadId: string): Promise<CodexHistoryEntry[] | undefined>;
}

export type CodexResponse =
	| { result: unknown }
	| { error: { code: number; message: string } };

/** A notification or request from the app-server. */
export interface CodexServerMessage {
	method: string;
	params: any;
	/** Set on requests, which wait on an answer. */
	id?: CodexRequestId;
}

/** What the host sends when its app-server stops: every thread is unloaded. */
export const CLOSED_METHOD = "host/closed";

/** One entry of a saved thread's history. */
export type CodexHistoryEntry =
	/** A turn started. */
	| { type: "turn"; turnId: string }
	/** Something the model saw or said. */
	| { type: "item"; turnId?: string; item: CodexResponseItem }
	/** A Responses API call ended, with all it said before this. */
	| { type: "response"; turnId: string; responseId: string }
	/** The history was compacted: everything before is replaced by `items`. */
	| { type: "compacted"; turnId?: string; items?: CodexResponseItem[] };

export interface CodexStatusOptions {
	/** The `codex` to ask, in place of the one the host finds. */
	executable?: string;
}

/** One of the models Codex offers the signed-in account. */
export interface CodexModelInfo {
	/** What to pass as the model. */
	value: string;
	displayName?: string;
	description?: string;
	isDefault?: boolean;
	efforts?: CodexReasoningEffort[];
	defaultEffort?: CodexReasoningEffort;
	/** Input it takes besides text, such as `image`. */
	inputModalities?: string[];
	/** Service tiers it offers, such as fast mode's. */
	serviceTiers?: string[];
}

export interface CodexStatus {
	installed: boolean;
	loggedIn: boolean;
	version?: string;
	/** `chatgpt` (a ChatGPT plan) or `apiKey`. */
	authMode?: string;
	planType?: string;
	error?: string;
	/** What the signed-in `codex` offers, once it has been asked. */
	models?: CodexModelInfo[];
}
