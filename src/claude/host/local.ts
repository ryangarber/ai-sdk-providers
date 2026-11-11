import { execFile } from "node:child_process";
import { open } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import {
	type CanUseTool,
	createSdkMcpServer,
	getSessionInfo,
	type HookCallback,
	type HookCallbackMatcher,
	type HookEvent,
	type HookInput,
	type Options,
	type Query,
	query,
	type SDKUserMessage,
	tool,
} from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";
import { findOnPath, isExecutable } from "../../shared/executable.ts";
import type {
	ClaudeCodeHost,
	ClaudeCodeHostAnswer,
	ClaudeCodeHostEvent,
	ClaudeCodeModelInfo,
	ClaudeCodeStatus,
	ClaudeCodeToolDefinition,
} from "./protocol.ts";

type Env = Record<string, string | undefined>;
/** Something a run asks, before it has an id. */
type Question = ClaudeCodeHostEvent extends infer E
	? E extends { type: "tool-call" | "permission"; id: string }
		? Omit<E, "id">
		: never
	: never;
type Hooks = Partial<Record<HookEvent, HookCallbackMatcher[]>>;

/** What a host's hooks get to work with, for one run. */
export interface ClaudeCodeHookContext {
	/** The session's working directory. */
	cwd: string;
	/** `cwd` and the session's additional directories. */
	folders: string[];
}

export interface ClaudeCodeLocalHostSettings {
	/** The `claude` to run. Defaults to the first on `PATH`, then its installer's. */
	executable?: string;
	/**
	 * The environment Claude Code runs in, or how to get it. Defaults to this
	 * process's. `CLAUDECODE` is always dropped, so it runs inside a session.
	 */
	env?: Env | (() => Env | Promise<Env>);
	/**
	 * `subscription` (default) drops `ANTHROPIC_API_KEY`, so Claude Code runs on
	 * the signed-in account rather than billing a key it finds; `inherit` keeps it.
	 */
	auth?: "subscription" | "inherit";
	/**
	 * Where a session's project settings, agents and skills come from, for its
	 * `cwd`. Defaults to the main checkout when `cwd` is in a linked git
	 * worktree. `false` leaves it to Claude Code.
	 */
	projectConfigRoot?:
		| false
		| ((cwd: string) => string | undefined | Promise<string | undefined>);
	/** Agent SDK hooks for every run, after the host's own. */
	hooks?: Hooks | ((context: ClaudeCodeHookContext) => Hooks);
	/** Claude Code's stderr, for debugging. */
	stderr?: (data: string) => void;
}

/** How long a status is trusted, since it is asked for before every turn. */
const STATUS_MS = { valid: 60_000, invalid: 5_000 };
/**
 * How long what `claude` offers is trusted. Asking starts it, so it is asked
 * again only this rarely, or once its version or sign-in changes.
 */
const OFFER_MS = 30 * 60_000;
const OFFER_TIMEOUT_MS = 20_000;

/** Where Claude Code's installer puts `claude`, beyond `PATH`. */
const KNOWN_PATHS = [
	join(homedir(), ".local", "bin", "claude"),
	join(homedir(), ".claude", "local", "claude"),
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

/** A JSON schema as the zod shape the SDK takes tools' inputs as. */
const toShape = (schema: unknown) => {
	try {
		const parsed = z.fromJSONSchema(schema as any);
		if (parsed instanceof z.ZodObject) return parsed.shape;
	} catch {}
	return {};
};

/** The main checkout `cwd` is a linked worktree of, if it is one. */
const getWorktreeMain = async (cwd: string, env: Env) => {
	try {
		const output = await exec(
			"git",
			[
				"-C",
				cwd,
				"rev-parse",
				"--path-format=absolute",
				"--git-common-dir",
				"--show-toplevel",
			],
			env,
		);
		const [common, top] = output
			.trim()
			.split(/\r?\n/)
			.map((path) => path.trim().replace(/(?<=.)\/+$/, ""));
		if (!common?.endsWith("/.git") || !top) return undefined;
		const main = common.slice(0, -"/.git".length) || "/";
		return main === top ? undefined : main;
	} catch {
		return undefined;
	}
};

/** An async queue of a session's input, open until the run ends. */
const createInput = () => {
	const messages: SDKUserMessage[] = [];
	let wake: (() => void) | undefined;
	let closed = false;
	return {
		push: (message: SDKUserMessage) => {
			if (closed) return;
			messages.push(message);
			wake?.();
		},
		close: () => {
			closed = true;
			wake?.();
		},
		messages: (async function* () {
			while (true) {
				while (messages.length) yield messages.shift() as SDKUserMessage;
				if (closed) return;
				await new Promise<void>((resolve) => (wake = resolve));
				wake = undefined;
			}
		})(),
	};
};

interface Run {
	abort: AbortController;
	input: ReturnType<typeof createInput>;
	query?: Query;
	/** What the run asked and waits on, by question id. */
	questions: Map<string, (answer: ClaudeCodeHostAnswer) => void>;
	/** Subagents and background work, by the call that started them. */
	tasks: Map<string, string>;
	mcpServerName: string;
	onToolCall: (
		definition: ClaudeCodeToolDefinition,
		input: unknown,
		extra: unknown,
	) => Promise<{
		content: { type: "text"; text: string }[];
		isError?: boolean;
	}>;
}

/**
 * Claude Code in this process, through the Agent SDK, on the installed
 * `claude` and its own sign-in. Node and Bun only.
 */
export function createLocalHost(
	settings: ClaudeCodeLocalHostSettings = {},
): ClaudeCodeHost {
	const runs = new Map<string, Run>();

	const getEnv = async (): Promise<Env> => {
		const base =
			typeof settings.env === "function"
				? await settings.env()
				: (settings.env ?? process.env);
		const env = { ...base };
		if ((settings.auth ?? "subscription") === "subscription")
			delete env.ANTHROPIC_API_KEY;
		delete env.CLAUDECODE;
		return env;
	};

	let found: Promise<string | undefined> | undefined;
	const findExecutable = async (chosen = settings.executable?.trim()) => {
		if (chosen) return (await isExecutable(chosen)) ? chosen : undefined;
		found ??= (async () => {
			return await findOnPath("claude", await getEnv(), KNOWN_PATHS);
		})();
		const file = await found;
		// Looked for again next time: it may be installed meanwhile.
		if (!file) found = undefined;
		return file;
	};

	let status:
		| { value: ClaudeCodeStatus; at: number; executable?: string }
		| undefined;
	let offer:
		| { value: Partial<ClaudeCodeStatus>; at: number; key: string }
		| undefined;

	/** What the signed-in `claude` offers, from a session started only to ask. */
	const getOffer = async (file: string, key: string) => {
		if (offer?.key === key && Date.now() - offer.at < OFFER_MS)
			return offer.value;
		const abort = new AbortController();
		const timeout = setTimeout(() => abort.abort(), OFFER_TIMEOUT_MS);
		let run: Query | undefined;
		try {
			run = query({
				prompt: (async function* (): AsyncGenerator<SDKUserMessage> {
					await new Promise((resolve) =>
						abort.signal.addEventListener("abort", resolve),
					);
				})(),
				options: {
					cwd: homedir(),
					env: await getEnv(),
					pathToClaudeCodeExecutable: file,
					abortController: abort,
					persistSession: false,
					settingSources: ["user"],
				},
			});
			const result = await Promise.race([
				run.initializationResult(),
				new Promise<never>((_, reject) =>
					abort.signal.addEventListener("abort", () =>
						reject(new Error("Timed out")),
					),
				),
			]);
			const value: Partial<ClaudeCodeStatus> = {
				models: result.models.map(
					(model): ClaudeCodeModelInfo => ({
						value: model.value,
						displayName: model.displayName,
						description: model.description,
						efforts: model.supportsEffort
							? model.supportedEffortLevels
							: undefined,
						adaptiveThinking: model.supportsAdaptiveThinking,
						fastMode: model.supportsFastMode,
						autoMode: model.supportsAutoMode,
					}),
				),
				agents: result.agents.map((agent) => agent.name),
				outputStyles: result.available_output_styles,
				outputStyle: result.output_style,
			};
			offer = { value, at: Date.now(), key };
			return value;
		} catch {
			return offer?.key === key ? offer.value : {};
		} finally {
			clearTimeout(timeout);
			abort.abort();
			run?.close();
		}
	};

	const createMcpServer = (run: Run, tools: ClaudeCodeToolDefinition[]) =>
		createSdkMcpServer({
			name: run.mcpServerName,
			alwaysLoad: true,
			tools: tools.map((definition) =>
				tool(
					definition.name,
					definition.description ?? "",
					toShape(definition.inputSchema),
					(input, extra) => run.onToolCall(definition, input, extra),
				),
			),
		});

	const getRun = (id: string) => {
		const run = runs.get(id);
		if (!run) throw new Error(`No Claude Code run ${id}`);
		return run;
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

			let value: ClaudeCodeStatus;
			const file = await findExecutable(chosen);
			if (!file) value = { installed: false, loggedIn: false };
			else {
				try {
					const env = await getEnv();
					const [version, auth] = await Promise.all([
						exec(file, ["--version"], env),
						exec(file, ["auth", "status"], env).catch(
							(error) => error?.stdout ?? "{}",
						),
					]);
					value = {
						installed: true,
						loggedIn: JSON.parse(auth || "{}").loggedIn === true,
						version: version.trim().split(" ")[0],
					};
					if (value.loggedIn)
						Object.assign(
							value,
							await getOffer(file, `${file}@${value.version}`),
						);
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

		async start(request, onEvent) {
			const id = crypto.randomUUID();
			const abort = new AbortController();
			const input = createInput();
			const questions = new Map<
				string,
				(answer: ClaudeCodeHostAnswer) => void
			>();

			/** Waits on an answer, or a refusal once the run ends. */
			const ask = (event: Question, fallback: ClaudeCodeHostAnswer) =>
				new Promise<ClaudeCodeHostAnswer>((resolve) => {
					if (abort.signal.aborted) return resolve(fallback);
					const question = crypto.randomUUID();
					questions.set(question, resolve);
					abort.signal.addEventListener("abort", () => resolve(fallback), {
						once: true,
					});
					onEvent({ ...event, id: question } as ClaudeCodeHostEvent);
				});

			const run: Run = {
				abort,
				input,
				questions,
				tasks: new Map(),
				mcpServerName: request.mcpServerName,
				async onToolCall(definition, toolInput, extra) {
					const meta = (extra as { _meta?: Record<string, unknown> })?._meta;
					const answer = await ask(
						{
							type: "tool-call",
							name: definition.name,
							input: toolInput,
							toolUseId: meta?.["claudecode/toolUseId"] as string | undefined,
						},
						{
							type: "tool-result",
							content: [{ type: "text", text: "The tool did not run." }],
							isError: true,
						},
					);
					if (answer.type !== "tool-result")
						return {
							content: [{ type: "text", text: "The tool did not run." }],
							isError: true,
						};
					return {
						content: answer.content as { type: "text"; text: string }[],
						isError: answer.isError,
					};
				},
			};
			runs.set(id, run);

			const options = request.options;
			const cwd = options.cwd ?? homedir();
			const env = await getEnv();
			const prefix = `mcp__${request.mcpServerName}__`;

			const canUseTool: CanUseTool = async (name, toolInput, extra) => {
				const answer = await ask(
					{
						type: "permission",
						toolUseId: extra.toolUseID,
						name,
						input: toolInput,
						title: extra.title,
						displayName: extra.displayName,
						description: extra.description,
						decisionReason: extra.decisionReason,
						blockedPath: extra.blockedPath,
						agentId: extra.agentID,
					},
					{
						type: "permission",
						result: { behavior: "deny", message: "The run ended." },
					},
				);
				if (answer.type !== "permission")
					return { behavior: "deny", message: "No answer." };
				return answer.result.behavior === "allow"
					? {
							behavior: "allow",
							updatedInput: answer.result.updatedInput ?? toolInput,
						}
					: answer.result;
			};

			const call = (input: HookInput) =>
				"tool_use_id" in input && typeof input.tool_use_id === "string"
					? {
							toolUseId: input.tool_use_id,
							name: "tool_name" in input ? String(input.tool_name) : "",
							agentId: input.agent_id,
						}
					: undefined;

			const started: HookCallback = async (input) => {
				const found = call(input);
				if (found) onEvent({ type: "tool-start", ...found });
				// The caller's tools skip Claude Code's permission flow: the
				// caller approves them as it does its own.
				return found?.name.startsWith(prefix)
					? {
							hookSpecificOutput: {
								hookEventName: "PreToolUse",
								permissionDecision: "allow",
							},
						}
					: {};
			};
			const ended: HookCallback = async (input) => {
				const found = call(input);
				if (found)
					onEvent({
						type: "tool-end",
						toolUseId: found.toolUseId,
						agentId: found.agentId,
					});
				return {};
			};

			const context: ClaudeCodeHookContext = {
				cwd,
				folders: [cwd, ...(options.additionalDirectories ?? [])],
			};
			const extra =
				typeof settings.hooks === "function"
					? settings.hooks(context)
					: (settings.hooks ?? {});
			const hooks: Hooks = {
				PreToolUse: [{ hooks: [started] }],
				PostToolUse: [{ hooks: [ended] }],
				PostToolUseFailure: [{ hooks: [ended] }],
				PermissionDenied: [{ hooks: [ended] }],
			};
			for (const [event, matchers] of Object.entries(extra) as [
				HookEvent,
				HookCallbackMatcher[],
			][])
				hooks[event] = [...(hooks[event] ?? []), ...matchers];

			const projectConfigRoot =
				options.projectConfigRoot ??
				(settings.projectConfigRoot === false
					? undefined
					: settings.projectConfigRoot
						? await settings.projectConfigRoot(cwd)
						: await getWorktreeMain(cwd, env));

			const mcpServers = {
				...options.mcpServers,
				...(request.tools.length
					? { [request.mcpServerName]: createMcpServer(run, request.tools) }
					: {}),
			};

			const sdkOptions: Options = {
				...options,
				cwd,
				projectConfigRoot,
				env,
				pathToClaudeCodeExecutable: await findExecutable(
					request.executable?.trim() || undefined,
				),
				abortController: abort,
				includePartialMessages: true,
				// A subagent's work streams inside its call.
				forwardSubagentText: true,
				mcpServers,
				hooks,
				canUseTool,
				allowDangerouslySkipPermissions:
					options.permissionMode === "bypassPermissions" || undefined,
				stderr: settings.stderr,
			};

			input.push(request.prompt);
			void (async () => {
				try {
					run.query = query({ prompt: input.messages, options: sdkOptions });
					/** Background shells' tasks, whose output only reaches a file. */
					const shells = new Set<string>();
					for await (const message of run.query) {
						if (
							message.type === "system" &&
							message.subtype === "task_started"
						) {
							if (message.tool_use_id)
								run.tasks.set(message.tool_use_id, message.task_id);
							if (message.task_type === "local_bash")
								shells.add(message.task_id);
						}
						if (
							message.type === "system" &&
							message.subtype === "task_notification" &&
							shells.delete(message.task_id) &&
							message.output_file
						) {
							const taskOutput = await readTail(message.output_file);
							onEvent({
								type: "message",
								message,
								...(taskOutput !== undefined && { taskOutput }),
							});
							continue;
						}
						onEvent({ type: "message", message });
					}
				} catch (error) {
					if (!abort.signal.aborted)
						onEvent({
							type: "error",
							message: error instanceof Error ? error.message : String(error),
						});
				} finally {
					input.close();
					abort.abort();
					runs.delete(id);
					onEvent({ type: "end" });
				}
			})();

			return id;
		},

		async send(id, message) {
			getRun(id).input.push(message);
		},

		async answer(id, question, answer) {
			const run = runs.get(id);
			const resolve = run?.questions.get(question);
			run?.questions.delete(question);
			resolve?.(answer);
		},

		async update(id, update) {
			const run = getRun(id);
			const live = run.query;
			if (!live) return;
			if (update.tools)
				await live.setMcpServers(
					update.tools.length
						? { [run.mcpServerName]: createMcpServer(run, update.tools) }
						: {},
				);
			if (update.model) await live.setModel(update.model);
			if (update.permissionMode)
				await live.setPermissionMode(update.permissionMode);
			if (update.settings)
				await live.applyFlagSettings(update.settings as never);
		},

		async interrupt(id) {
			await runs.get(id)?.query?.interrupt();
		},

		async stopTask(id, toolUseId) {
			const run = runs.get(id);
			const task = run?.tasks.get(toolUseId);
			if (task) await run?.query?.stopTask(task).catch(() => {});
		},

		async close(id) {
			const run = runs.get(id);
			if (!run) return;
			run.input.close();
			run.abort.abort();
		},

		async hasSession(session, options) {
			const info = await getSessionInfo(session, {
				dir: options?.cwd ?? homedir(),
			}).catch(() => undefined);
			return !!info;
		},
	};
}

/** The most of a task's output shown: its end, as a shell's last lines matter most. */
const TASK_OUTPUT_BYTES = 64 * 1024;

/** A file's end, up to {@link TASK_OUTPUT_BYTES}; undefined if unreadable. */
async function readTail(path: string): Promise<string | undefined> {
	const file = await open(path).catch(() => undefined);
	if (!file) return undefined;
	try {
		const { size } = await file.stat();
		const length = Math.min(size, TASK_OUTPUT_BYTES);
		const { buffer } = await file.read({
			buffer: Buffer.alloc(length),
			position: size - length,
		});
		const text = buffer.toString("utf8");
		// Cut mid-line (or mid-character): start at the next whole line.
		return size > length ? text.slice(text.indexOf("\n") + 1) : text;
	} catch {
		return undefined;
	} finally {
		await file.close();
	}
}
