import { type ChildProcess, execFile, spawn } from "node:child_process";
import { access, constants, readdir, readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { delimiter, join } from "node:path";
import { createInterface } from "node:readline";
import type {
	CodexModel,
	CodexResponseItem,
	CodexThreadResponse,
} from "./app-server.ts";
import {
	CLOSED_METHOD,
	type CodexHistoryEntry,
	type CodexHost,
	type CodexModelInfo,
	type CodexRequestId,
	type CodexServerMessage,
	type CodexStatus,
} from "./protocol.ts";

type Env = Record<string, string | undefined>;

export interface CodexLocalHostSettings {
	/** The `codex` to run. Defaults to the first on `PATH`, then its installers'. */
	executable?: string;
	/**
	 * The environment Codex runs in, or how to get it. Defaults to this
	 * process's.
	 */
	env?: Env | (() => Env | Promise<Env>);
	/** More arguments for `codex app-server`, such as `-c key=value`. */
	args?: string[];
	/** How the app-server knows its client. */
	clientInfo?: { name: string; title?: string; version: string };
	/**
	 * Notifications the app-server doesn't send, to save the traffic. Defaults
	 * to a few that nothing here reads.
	 */
	optOutNotifications?: string[];
	/** Codex's stderr, for debugging. */
	stderr?: (data: string) => void;
}

/** How long a status is trusted, since it may be asked for before every turn. */
const STATUS_MS = { valid: 60_000, invalid: 5_000 };
/** How long the models offered are trusted. */
const MODELS_MS = 30 * 60_000;

const DEFAULT_OPT_OUT = [
	"account/rateLimits/updated",
	"mcpServer/startupStatus/updated",
	"remoteControl/status/changed",
	"thread/readState/changed",
	"turn/moderationMetadata",
];

/** Where Codex's installers put `codex`, beyond `PATH`. */
const KNOWN_PATHS = [
	join(homedir(), ".local", "bin", "codex"),
	"/opt/homebrew/bin/codex",
	"/usr/local/bin/codex",
];

const exec = (file: string, args: string[], env: Env) =>
	new Promise<string>((resolve, reject) =>
		execFile(
			file,
			args,
			{ env: env as NodeJS.ProcessEnv, timeout: 15_000 },
			(error, stdout) => (error ? reject(error) : resolve(stdout)),
		),
	);

const isExecutable = async (file: string) => {
	try {
		await access(file, constants.X_OK);
		return true;
	} catch {
		return false;
	}
};

class RpcError extends Error {
	readonly code?: number;
	readonly data?: unknown;
	constructor(error: { code?: number; message?: string; data?: unknown }) {
		super(error.message ?? "Codex app-server error");
		this.name = "CodexRpcError";
		this.code = error.code;
		this.data = error.data;
	}
}

/** One running `codex app-server`, spoken to over stdio. */
interface Server {
	child: ChildProcess;
	ready: Promise<{ codexHome?: string }>;
	pending: Map<
		CodexRequestId,
		{ resolve: (value: unknown) => void; reject: (error: Error) => void }
	>;
	next: number;
	closed: boolean;
}

/**
 * Codex in this process's machine: the installed `codex`, its app-server
 * started on first use and its own sign-in. Node and Bun only.
 */
export function createLocalHost(
	settings: CodexLocalHostSettings = {},
): CodexHost {
	const listeners = new Set<(message: CodexServerMessage) => void>();
	const emit = (message: CodexServerMessage) => {
		for (const listener of listeners) listener(message);
	};

	const getEnv = async (): Promise<Env> => {
		const base =
			typeof settings.env === "function"
				? await settings.env()
				: (settings.env ?? process.env);
		return { ...base };
	};

	let found: Promise<string | undefined> | undefined;
	const findExecutable = async (chosen = settings.executable?.trim()) => {
		if (chosen) return (await isExecutable(chosen)) ? chosen : undefined;
		found ??= (async () => {
			const env = await getEnv();
			const name = process.platform === "win32" ? "codex.exe" : "codex";
			const candidates = [
				...(env.PATH ?? "")
					.split(delimiter)
					.filter(Boolean)
					.map((dir) => join(dir, name)),
				...KNOWN_PATHS,
			];
			for (const candidate of candidates)
				if (await isExecutable(candidate)) return candidate;
			return undefined;
		})();
		const file = await found;
		// Looked for again next time: it may be installed meanwhile.
		if (!file) found = undefined;
		return file;
	};

	let server: Server | undefined;
	let codexHome: string | undefined;

	const write = (target: Server, message: unknown) => {
		target.child.stdin?.write(`${JSON.stringify(message)}\n`);
	};

	const send = (target: Server, method: string, params: unknown) =>
		new Promise<unknown>((resolve, reject) => {
			if (target.closed)
				return reject(new Error("The Codex app-server has stopped."));
			const id = target.next++;
			target.pending.set(id, { resolve, reject });
			write(target, { id, method, params });
		});

	const start = async (): Promise<Server> => {
		const file = await findExecutable();
		if (!file) throw new Error("Codex isn't installed: no `codex` was found.");
		const child = spawn(file, ["app-server", ...(settings.args ?? [])], {
			env: (await getEnv()) as NodeJS.ProcessEnv,
			stdio: ["pipe", "pipe", "pipe"],
		});
		const target: Server = {
			child,
			ready: Promise.resolve({}),
			pending: new Map(),
			next: 1,
			closed: false,
		};
		child.stderr?.on("data", (data) => settings.stderr?.(String(data)));
		createInterface({ input: child.stdout as NodeJS.ReadableStream }).on(
			"line",
			(line) => {
				let message: any;
				try {
					message = JSON.parse(line);
				} catch {
					return;
				}
				if (message.method === undefined && message.id !== undefined) {
					const call = target.pending.get(message.id);
					target.pending.delete(message.id);
					if (message.error) call?.reject(new RpcError(message.error));
					else call?.resolve(message.result);
					return;
				}
				if (typeof message.method === "string")
					emit({
						method: message.method,
						params: message.params,
						...(message.id !== undefined && { id: message.id }),
					});
			},
		);
		const stop = (error: Error) => {
			if (target.closed) return;
			target.closed = true;
			if (server === target) server = undefined;
			for (const call of target.pending.values()) call.reject(error);
			target.pending.clear();
			emit({ method: CLOSED_METHOD, params: { message: error.message } });
		};
		child.on("error", (error) => stop(error));
		child.on("exit", (code, signal) =>
			stop(
				new Error(
					`The Codex app-server stopped (${signal ?? `exit code ${code}`}).`,
				),
			),
		);
		const info = settings.clientInfo ?? {
			name: "ai_sdk_providers",
			version: "0.1.0",
		};
		target.ready = (async () => {
			const result = (await send(target, "initialize", {
				clientInfo: {
					name: info.name,
					title: info.title ?? null,
					version: info.version,
				},
				capabilities: {
					experimentalApi: true,
					requestAttestation: false,
					optOutNotificationMethods:
						settings.optOutNotifications ?? DEFAULT_OPT_OUT,
				},
			})) as { codexHome?: string };
			write(target, { method: "initialized", params: {} });
			codexHome = result.codexHome;
			return result;
		})();
		target.ready.catch(() => target.child.kill());
		return target;
	};

	let starting: Promise<Server> | undefined;
	const getServer = async () => {
		if (server && !server.closed) {
			await server.ready;
			return server;
		}
		starting ??= start().finally(() => {
			starting = undefined;
		});
		const target = await starting;
		server = target;
		await target.ready;
		return target;
	};

	const request = async (method: string, params: unknown) =>
		send(await getServer(), method, params);

	let status:
		| { value: CodexStatus; at: number; executable?: string }
		| undefined;
	let models: { value: CodexModelInfo[]; at: number } | undefined;

	const getModels = async () => {
		if (models && Date.now() - models.at < MODELS_MS) return models.value;
		const value: CodexModelInfo[] = [];
		let cursor: string | null | undefined;
		do {
			const page = (await request("model/list", { cursor, limit: 100 })) as {
				data: CodexModel[];
				nextCursor?: string | null;
			};
			for (const model of page.data)
				value.push({
					value: model.model ?? model.id,
					displayName: model.displayName,
					description: model.description,
					isDefault: model.isDefault,
					efforts: model.supportedReasoningEfforts.map(
						(effort) => effort.reasoningEffort,
					),
					defaultEffort: model.defaultReasoningEffort,
					inputModalities: model.inputModalities,
					serviceTiers: model.serviceTiers?.map((tier) => tier.id),
				});
			cursor = page.nextCursor;
		} while (cursor);
		models = { value, at: Date.now() };
		return value;
	};

	/** Where a rollout is saved, by its id, under the Codex home. */
	const findRollout = async (id: string) => {
		const home = codexHome ?? join(homedir(), ".codex");
		for (const dir of ["sessions", "archived_sessions"]) {
			try {
				const files = await readdir(join(home, dir), { recursive: true });
				const file = files.find((name) => name.endsWith(`-${id}.jsonl`));
				if (file) return join(home, dir, file);
			} catch {}
		}
		return undefined;
	};

	/** A rollout's model-visible history, with the history it was forked from. */
	const readRollout = async (
		path: string,
		end?: number,
		depth = 0,
	): Promise<CodexHistoryEntry[]> => {
		const buffer = await readFile(path);
		const text = buffer.subarray(0, end ?? buffer.length).toString("utf8");
		const entries: CodexHistoryEntry[] = [];
		let turnId: string | undefined;
		let sawMeta = false;
		for (const line of text.split("\n")) {
			if (!line.trim()) continue;
			let record: { type?: string; payload?: any };
			try {
				record = JSON.parse(line);
			} catch {
				continue;
			}
			const payload = record.payload ?? {};
			switch (record.type) {
				case "session_meta": {
					if (sawMeta) break;
					sawMeta = true;
					const base = payload.history_base;
					if (base?.thread_id && depth < 32) {
						const prefix = await findRollout(base.thread_id);
						if (!prefix) throw new Error(`No rollout ${base.thread_id}`);
						entries.push(
							...(await readRollout(prefix, base.end_byte_offset, depth + 1)),
						);
					}
					break;
				}
				case "turn_context":
				case "event_msg": {
					const started =
						record.type === "turn_context"
							? payload.turn_id
							: payload.type === "task_started"
								? (payload.turn_attribution?.turn_id ?? payload.turn_id)
								: undefined;
					if (typeof started === "string" && started !== turnId) {
						turnId = started;
						entries.push({ type: "turn", turnId });
					}
					break;
				}
				case "response_item":
					entries.push({
						type: "item",
						turnId,
						item: payload as CodexResponseItem,
					});
					break;
				case "token_usage_record":
					if (typeof payload.response_id === "string")
						entries.push({
							type: "response",
							turnId: payload.turn_id ?? turnId,
							responseId: payload.response_id,
						});
					break;
				case "compacted":
					entries.push({
						type: "compacted",
						turnId,
						items: (payload.replacement_history ?? undefined) as
							| CodexResponseItem[]
							| undefined,
					});
					break;
			}
		}
		return entries;
	};

	return {
		async status(options) {
			const chosen = options?.executable?.trim() || settings.executable;
			if (
				status &&
				status.executable === chosen &&
				Date.now() - status.at <
					(status.value.loggedIn ? STATUS_MS.valid : STATUS_MS.invalid)
			)
				return status.value;

			let value: CodexStatus;
			const file = await findExecutable(chosen);
			if (!file) value = { installed: false, loggedIn: false };
			else {
				try {
					const version = (await exec(file, ["--version"], await getEnv()))
						.trim()
						.split(/\s+/)
						.at(-1);
					const account = (await request("account/read", {})) as {
						account: { type: string; planType?: string } | null;
						requiresOpenaiAuth: boolean;
					};
					value = {
						installed: true,
						loggedIn: !!account.account || !account.requiresOpenaiAuth,
						version,
						authMode: account.account?.type,
						planType: account.account?.planType,
					};
					if (value.loggedIn)
						value.models = await getModels().catch(() => undefined);
				} catch (error) {
					value = {
						installed: true,
						loggedIn: false,
						error: error instanceof Error ? error.message : String(error),
					};
				}
			}
			status = { value, at: Date.now(), executable: chosen };
			return value;
		},

		request,

		async respond(id, response) {
			const target = await getServer();
			write(target, { id, ...response });
		},

		listen(handler) {
			listeners.add(handler);
			return () => listeners.delete(handler);
		},

		async history(threadId) {
			try {
				const { thread } = (await request("thread/read", {
					threadId,
				})) as CodexThreadResponse;
				const path = thread.path ?? (await findRollout(threadId));
				if (!path) return undefined;
				return await readRollout(path);
			} catch {
				return undefined;
			}
		},
	};
}
