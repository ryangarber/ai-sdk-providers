import { homedir } from "node:os";
import {
	Agent,
	type AgentOptions,
	Cursor,
	FileCredentialStore,
	type Run,
	type SDKAgent,
	type SDKCustomTool,
	type SDKCustomToolResult,
} from "@cursor/sdk";
import type {
	CursorAgentOptions,
	CursorHost,
	CursorHostEvent,
	CursorModelInfo,
	CursorStatus,
	CursorToolAnswer,
	CursorToolDefinition,
} from "./protocol.ts";

export interface CursorLocalHostSettings {
	/**
	 * The API key agents run on, or how to get it: one minted by
	 * `Cursor.auth.login()`, or a user or service account key. Defaults to
	 * `CURSOR_API_KEY`, then the key `Cursor.auth.login()` saved by default.
	 */
	apiKey?: string | (() => string | undefined | Promise<string | undefined>);
	/** SDK agent options for every agent, under the provider's own. */
	agentOptions?: Partial<Omit<AgentOptions, "local" | "cloud">> & {
		local?: Partial<NonNullable<AgentOptions["local"]>>;
	};
}

/** How long the account's models are trusted. */
const MODELS_MS = 30 * 60_000;

const INTERRUPTED: CursorToolAnswer = {
	content: [{ type: "text", text: "Interrupted." }],
	isError: true,
};

interface Live {
	agent: SDKAgent;
	/** What it was opened with, to tell when it has to be reopened. */
	key: string;
	run?: Run;
	/** The caller's tool calls waiting on an answer, by call id. */
	pending: Map<string, (answer: CursorToolAnswer) => void>;
}

const toResult = (answer: CursorToolAnswer): SDKCustomToolResult => ({
	content: answer.content.map((part) =>
		part.type === "text"
			? { type: "text" as const, text: part.text }
			: { type: "image" as const, data: part.data, mimeType: part.mimeType },
	),
	...(answer.isError && { isError: true }),
});

/**
 * Cursor's local agents in this process, through the Cursor SDK, on an API
 * key. Node and Bun only.
 */
export function createLocalHost(
	settings: CursorLocalHostSettings = {},
): CursorHost {
	const agents = new Map<string, Live>();

	const getApiKey = async () => {
		const chosen =
			typeof settings.apiKey === "function"
				? await settings.apiKey()
				: settings.apiKey;
		if (chosen) return chosen;
		if (process.env.CURSOR_API_KEY) return process.env.CURSOR_API_KEY;
		const stored = await new FileCredentialStore().load().catch(() => {});
		if (
			stored &&
			(!stored.apiKeyExpiresAtMs || stored.apiKeyExpiresAtMs > Date.now())
		)
			return stored.apiKey;
		return undefined;
	};

	const requireApiKey = async () => {
		const apiKey = await getApiKey();
		if (!apiKey)
			throw new Error(
				"No Cursor API key: pass one, set CURSOR_API_KEY, or sign in with Cursor.auth.login().",
			);
		return apiKey;
	};

	let models: { value: CursorModelInfo[]; at: number; key: string } | undefined;

	const toAgentOptions = async (
		options: CursorAgentOptions,
	): Promise<AgentOptions> => {
		const base = settings.agentOptions ?? {};
		const local = {
			...base.local,
			cwd: options.cwd ?? base.local?.cwd ?? homedir(),
			...(options.dirs?.length && { dirs: options.dirs }),
			...(options.settingSources && {
				settingSources: options.settingSources,
			}),
			...(options.sandbox !== undefined && {
				sandboxOptions: { enabled: options.sandbox },
			}),
			...(options.autoReview !== undefined && {
				autoReview: options.autoReview,
			}),
		};
		const agentOptions: AgentOptions = {
			...base,
			apiKey: await requireApiKey(),
			model: options.model,
			local,
		};
		if (options.tools) agentOptions.tools = options.tools;
		if (options.disallowedTools?.length)
			agentOptions.disallowedTools = options.disallowedTools;
		if (options.systemPrompt) agentOptions.systemPrompt = options.systemPrompt;
		if (options.mcpServers) agentOptions.mcpServers = options.mcpServers;
		if (options.agents)
			agentOptions.agents = options.agents as AgentOptions["agents"];
		if (options.mode) agentOptions.mode = options.mode;
		if (options.name) agentOptions.name = options.name;
		return agentOptions;
	};

	const getLive = (agentId: string) => {
		const live = agents.get(agentId);
		if (!live) throw new Error(`No open Cursor agent ${agentId}`);
		return live;
	};

	const settle = (live: Live) => {
		for (const resolve of live.pending.values()) resolve(INTERRUPTED);
		live.pending.clear();
	};

	const customTools = (
		live: Live,
		tools: CursorToolDefinition[],
		onEvent: (event: CursorHostEvent) => void,
	) =>
		Object.fromEntries(
			tools.map((definition): [string, SDKCustomTool] => [
				definition.name,
				{
					description: definition.description ?? "",
					inputSchema: definition.inputSchema as SDKCustomTool["inputSchema"],
					execute: async (input, context) => {
						const toolCallId = context.toolCallId ?? crypto.randomUUID();
						const answer = await new Promise<CursorToolAnswer>((resolve) => {
							live.pending.set(toolCallId, resolve);
							onEvent({
								type: "tool-call",
								toolCallId,
								name: definition.name,
								input,
							});
						});
						return toResult(answer);
					},
				},
			]),
		);

	return {
		async status(): Promise<CursorStatus> {
			const apiKey = await getApiKey();
			if (!apiKey) return { loggedIn: false };
			if (models?.key === apiKey && Date.now() - models.at < MODELS_MS)
				return { loggedIn: true, models: models.value };
			try {
				const list = await Cursor.models.list({ apiKey });
				const value = list.map(
					(model): CursorModelInfo => JSON.parse(JSON.stringify(model)),
				);
				models = { value, at: Date.now(), key: apiKey };
				return { loggedIn: true, models: value };
			} catch (error) {
				return {
					loggedIn: false,
					error: error instanceof Error ? error.message : String(error),
				};
			}
		},

		async open(request) {
			const key = JSON.stringify(request.options);
			const id = request.agentId;
			const live = id ? agents.get(id) : undefined;
			if (live?.key === key) return live.agent.agentId;
			if (live) {
				if (live.run && live.run.status === "running")
					throw new Error(`Cursor agent ${id} is busy`);
				agents.delete(live.agent.agentId);
				live.agent.close();
			}
			const options = await toAgentOptions(request.options);
			const agent = id
				? await Agent.resume(id, options)
				: await Agent.create(options);
			agents.set(agent.agentId, { agent, key, pending: new Map() });
			return agent.agentId;
		},

		async send(agentId, request, onEvent) {
			const live = getLive(agentId);
			const run = await live.agent.send(request.message, {
				...(request.model && { model: request.model }),
				...(request.mode && { mode: request.mode }),
				local: { customTools: customTools(live, request.tools, onEvent) },
				onDelta: ({ update }) => onEvent({ type: "delta", update }),
			});
			live.run = run;
			void (async () => {
				try {
					const result = await run.wait();
					onEvent({
						type: "result",
						status: result.status,
						error: result.error?.message,
						usage: result.usage,
						model: result.model,
						durationMs: result.durationMs,
						requestId: result.requestId,
					});
				} catch (error) {
					onEvent({
						type: "error",
						message: error instanceof Error ? error.message : String(error),
					});
				} finally {
					if (live.run === run) live.run = undefined;
					settle(live);
					onEvent({ type: "end" });
				}
			})();
			return run.id;
		},

		async answer(agentId, toolCallId, answer) {
			const live = agents.get(agentId);
			const resolve = live?.pending.get(toolCallId);
			live?.pending.delete(toolCallId);
			resolve?.(answer);
		},

		async steer(agentId, text) {
			const run = agents.get(agentId)?.run;
			if (!run?.steer || run.status !== "running" || !run.supports("stream"))
				return false;
			try {
				return (await run.steer(text)) === "complete_delivered";
			} catch {
				return false;
			}
		},

		async cancel(agentId) {
			const live = agents.get(agentId);
			if (!live) return;
			settle(live);
			if (live.run?.status === "running")
				await live.run.cancel().catch(() => {});
		},

		async close(agentId) {
			const live = agents.get(agentId);
			if (!live) return;
			agents.delete(agentId);
			settle(live);
			if (live.run?.status === "running")
				await live.run.cancel().catch(() => {});
			live.agent.close();
		},

		async lastRun(agentId, options) {
			try {
				const { items } = await Agent.listRuns(agentId, {
					runtime: "local",
					...(options?.cwd && { cwd: options.cwd }),
				});
				const last = items
					.slice()
					.sort((a, b) => (b.createdAt ?? 0) - (a.createdAt ?? 0))[0];
				return last ? { id: last.id, status: last.status } : undefined;
			} catch {
				return undefined;
			}
		},
	};
}
