import type { JSONObject } from "@ai-sdk/provider";
import type {
	EffortLevel,
	McpHttpServerConfig,
	McpSSEServerConfig,
	McpStdioServerConfig,
	Options,
	PermissionMode,
	SDKMessage,
	SDKUserMessage,
} from "@anthropic-ai/claude-agent-sdk";

/**
 * How a provider reaches Claude Code: everything here is JSON, so a host can
 * sit across any boundary — in-process ({@link createLocalHost}), or in another
 * process or thread over a message channel (`serveClaudeCodeHost` /
 * `connectClaudeCodeHost`). The host is thin: it runs the Agent SDK and
 * relays what it says and asks; the provider does the rest.
 */
export interface ClaudeCodeHost {
	/** Whether `claude` is installed and signed in, and what it offers. */
	status(options?: ClaudeCodeStatusOptions): Promise<ClaudeCodeStatus>;
	/**
	 * Starts a session with its first message, returning the run's id. Events
	 * come through `onEvent` until an `end` event.
	 */
	start(
		request: ClaudeCodeRunRequest,
		onEvent: (event: ClaudeCodeHostEvent) => void,
	): Promise<string>;
	/** Sends a user message into a running session. */
	send(run: string, message: SDKUserMessage): Promise<void>;
	/** Answers something a run asked: a tool call or a permission. */
	answer(run: string, id: string, answer: ClaudeCodeHostAnswer): Promise<void>;
	/** Changes what a running session uses from its next step on. */
	update(run: string, update: ClaudeCodeRunUpdate): Promise<void>;
	/** Stops the turn in progress; the session stays open. */
	interrupt(run: string): Promise<void>;
	/** Stops one subagent of a run, by the call that started it. */
	stopTask(run: string, toolUseId: string): Promise<void>;
	/** Ends a run, whatever it is doing. */
	close(run: string): Promise<void>;
	/** Whether a session can be resumed from where it was saved. */
	hasSession(session: string, options?: { cwd?: string }): Promise<boolean>;
}

export interface ClaudeCodeStatusOptions {
	/** The `claude` to ask, in place of the one the host finds. */
	executable?: string;
}

/** One of the models Claude Code offers the signed-in account. */
export interface ClaudeCodeModelInfo {
	/** What to pass as the model: an alias such as `sonnet`, or a full id. */
	value: string;
	displayName?: string;
	description?: string;
	efforts?: EffortLevel[];
	adaptiveThinking?: boolean;
	fastMode?: boolean;
	autoMode?: boolean;
}

export interface ClaudeCodeStatus {
	installed: boolean;
	loggedIn: boolean;
	version?: string;
	error?: string;
	/** What the signed-in `claude` offers, once it has been asked. */
	models?: ClaudeCodeModelInfo[];
	/** Its agents, from the user's settings and Claude Code's own. */
	agents?: string[];
	outputStyles?: string[];
	/** The output style its settings pick. */
	outputStyle?: string;
}

/** An MCP server a session may use, configured as JSON. */
export type ClaudeCodeMcpServer =
	| McpStdioServerConfig
	| McpSSEServerConfig
	| McpHttpServerConfig;

/**
 * The Agent SDK's options that are plain data, as a host takes them. Anything
 * that is a function — hooks, `canUseTool`, `env` — is the host's own.
 */
export type ClaudeCodeRunOptions = Pick<
	Options,
	| "model"
	| "fallbackModel"
	| "effort"
	| "thinking"
	| "cwd"
	| "additionalDirectories"
	| "projectConfigRoot"
	| "systemPrompt"
	| "tools"
	| "allowedTools"
	| "disallowedTools"
	| "settingSources"
	| "settings"
	| "permissionMode"
	| "agent"
	| "agents"
	| "outputFormat"
	| "resume"
	| "resumeSessionAt"
	| "forkSession"
	| "persistSession"
	| "maxTurns"
	| "maxBudgetUsd"
	| "strictMcpConfig"
	| "skills"
	| "betas"
	| "sandbox"
	| "planModeInstructions"
	| "toolConfig"
> & {
	mcpServers?: Record<string, ClaudeCodeMcpServer>;
};

/** A tool of the provider's caller, which the host serves to Claude over MCP. */
export interface ClaudeCodeToolDefinition {
	name: string;
	description?: string;
	inputSchema: JSONObject;
}

export interface ClaudeCodeRunRequest {
	options: ClaudeCodeRunOptions;
	/** The `claude` to run, in place of the one the host finds. */
	executable?: string;
	/** The caller's tools, run through `tool-call` events. */
	tools: ClaudeCodeToolDefinition[];
	/** The MCP server the caller's tools are under: `mcp__<name>__<tool>`. */
	mcpServerName: string;
	/** The session's first message. */
	prompt: SDKUserMessage;
}

export interface ClaudeCodeRunUpdate {
	tools?: ClaudeCodeToolDefinition[];
	model?: string;
	permissionMode?: PermissionMode;
	/** Settings above all others, as Claude Code's `settings.json` takes them. */
	settings?: Record<string, unknown>;
}

/** The content of an MCP tool result. */
export type ClaudeCodeToolContent =
	| { type: "text"; text: string }
	| { type: "image"; data: string; mimeType: string };

export type ClaudeCodeHostEvent =
	| {
			type: "message";
			message: SDKMessage;
			/**
			 * With a background shell's `task_notification`: what its output
			 * file holds (the end of it, when long), which only the host can read.
			 */
			taskOutput?: string;
	  }
	/** One of the caller's tools was called; answer with a `tool-result`. */
	| {
			type: "tool-call";
			id: string;
			toolUseId?: string;
			name: string;
			input: unknown;
	  }
	/** Claude Code's permission flow asks; answer with a `permission`. */
	| {
			type: "permission";
			id: string;
			toolUseId: string;
			name: string;
			input: Record<string, unknown>;
			title?: string;
			displayName?: string;
			description?: string;
			decisionReason?: string;
			blockedPath?: string;
			agentId?: string;
	  }
	/** A tool is about to go through permissions and run. */
	| { type: "tool-start"; toolUseId: string; name: string; agentId?: string }
	/** A tool ran, failed, or was denied. */
	| { type: "tool-end"; toolUseId: string; agentId?: string }
	/** What a host's hook attached to a call, for its result. */
	/** The run failed outside of Claude Code's own messages. */
	| { type: "error"; message: string }
	/** No more events follow. */
	| { type: "end" };

export type ClaudeCodeHostAnswer =
	| { type: "tool-result"; content: ClaudeCodeToolContent[]; isError?: boolean }
	| {
			type: "permission";
			result:
				| { behavior: "allow"; updatedInput?: Record<string, unknown> }
				| { behavior: "deny"; message: string; interrupt?: boolean };
	  };
