import type { JSONObject } from "@ai-sdk/provider";
import type { InteractionUpdate } from "@cursor/sdk";

/**
 * How a provider reaches Cursor: everything here is JSON, so a host can sit
 * across any boundary — in-process ({@link createLocalHost}), or in another
 * process or thread over a message channel (`serveCursorHost` /
 * `connectCursorHost`). The host is thin: it runs the Cursor SDK's local
 * agents and relays what they say and ask; the provider does the rest.
 */
export interface CursorHost {
	/** Whether the host has a working API key, and the models it offers. */
	status(): Promise<CursorStatus>;
	/**
	 * Opens an agent: a new one, or a saved one by its id, with what it may
	 * use. An open agent with that id is reopened with them, between runs.
	 */
	open(request: CursorOpenRequest): Promise<string>;
	/**
	 * Starts a run of an open agent with a message, returning the run's id.
	 * Events come through `onEvent` until an `end` event.
	 */
	send(
		agentId: string,
		request: CursorSendRequest,
		onEvent: (event: CursorHostEvent) => void,
	): Promise<string>;
	/** Answers one of the caller's tools a run called, by the call's id. */
	answer(
		agentId: string,
		toolCallId: string,
		answer: CursorToolAnswer,
	): Promise<void>;
	/**
	 * Gives a run more of what the user said while it runs. False if the run
	 * couldn't take it, and it has to wait for the next.
	 */
	steer(agentId: string, text: string): Promise<boolean>;
	/** Cancels an agent's run in progress; the agent stays open. */
	cancel(agentId: string): Promise<void>;
	/** Closes an agent, whatever it is doing. It stays saved. */
	close(agentId: string): Promise<void>;
	/** A saved agent's last run, if there is one. */
	lastRun(
		agentId: string,
		options?: { cwd?: string },
	): Promise<{ id: string; status: string } | undefined>;
}

export type CursorSettingSource =
	| "project"
	| "user"
	| "team"
	| "mdm"
	| "plugins"
	| "all";

export type CursorMcpServer =
	| {
			type?: "stdio";
			command: string;
			args?: string[];
			env?: Record<string, string>;
			cwd?: string;
	  }
	| {
			type?: "http" | "sse";
			url: string;
			headers?: Record<string, string>;
			auth?: { CLIENT_ID: string; CLIENT_SECRET?: string; scopes?: string[] };
	  };

export interface CursorAgentDefinition {
	description: string;
	prompt: string;
	model?: string;
	[key: string]: unknown;
}

/** One of a model's parameters, set: `{ id: "fast", value: "true" }`. */
export interface CursorModelParameter {
	id: string;
	value: string;
}

export interface CursorModelSelection {
	id: string;
	params?: CursorModelParameter[];
}

export type CursorUserImage =
	| { url: string }
	| { data: string; mimeType: string };

/** What a run is sent: text, and images. */
export interface CursorUserMessage {
	text: string;
	images?: CursorUserImage[];
}

/** One of the models the account is offered. */
export interface CursorModelInfo {
	id: string;
	displayName?: string;
	description?: string;
	aliases?: string[];
	/** Its parameters, and the values each takes. */
	parameters?: {
		id: string;
		displayName?: string;
		values: { value: string; displayName?: string }[];
	}[];
	/** Named sets of parameters, such as a fast or thinking variant. */
	variants?: {
		params: CursorModelParameter[];
		displayName: string;
		description?: string;
		isDefault?: boolean;
	}[];
}

export interface CursorStatus {
	/** Whether the host has an API key. */
	loggedIn: boolean;
	error?: string;
	/** The account's models, once they have been asked for. */
	models?: CursorModelInfo[];
}

/** The Cursor SDK's agent options that are plain data, as a host takes them. */
export interface CursorAgentOptions {
	model: CursorModelSelection;
	/** Built-in tools offered; `undefined` for Cursor's default set. */
	tools?: string[];
	disallowedTools?: string[];
	systemPrompt?: string;
	mcpServers?: Record<string, CursorMcpServer>;
	agents?: Record<string, CursorAgentDefinition>;
	mode?: "agent" | "plan";
	name?: string;
	cwd?: string;
	dirs?: string[];
	settingSources?: CursorSettingSource[];
	sandbox?: boolean;
	autoReview?: boolean;
}

export interface CursorOpenRequest {
	/** A saved agent to resume; a new one if not given. */
	agentId?: string;
	options: CursorAgentOptions;
}

/** A tool of the provider's caller, which the host serves to the agent. */
export interface CursorToolDefinition {
	name: string;
	description?: string;
	inputSchema: JSONObject;
}

export interface CursorSendRequest {
	message: CursorUserMessage;
	model?: CursorModelSelection;
	mode?: "agent" | "plan";
	/** The caller's tools, run through `tool-call` events. */
	tools: CursorToolDefinition[];
}

/** The content of a tool result. */
export type CursorToolContent =
	| { type: "text"; text: string }
	| { type: "image"; data: string; mimeType: string };

export interface CursorToolAnswer {
	content: CursorToolContent[];
	isError?: boolean;
}

/** Token counts, as the Cursor SDK reports a run's. */
export interface CursorTokenUsage {
	inputTokens: number;
	outputTokens: number;
	cacheReadTokens: number;
	cacheWriteTokens: number;
	reasoningTokens?: number;
}

export type CursorHostEvent =
	/** What the agent streams, as the SDK's `onDelta` has it. */
	| { type: "delta"; update: InteractionUpdate }
	/** One of the caller's tools was called; answer it by `toolCallId`. */
	| { type: "tool-call"; toolCallId: string; name: string; input: unknown }
	/** The run ended. */
	| {
			type: "result";
			status: "finished" | "error" | "cancelled";
			error?: string;
			usage?: CursorTokenUsage;
			model?: CursorModelSelection;
			durationMs?: number;
			requestId?: string;
	  }
	/** The run failed outside of what the agent said. */
	| { type: "error"; message: string }
	/** No more events follow. */
	| { type: "end" };
