import { access, constants, readFile, stat } from "node:fs/promises";
import { delimiter, dirname, join } from "node:path";

const windows = () => process.platform === "win32";

/**
 * `PATH` from an environment, whatever its case. Windows calls it `Path`,
 * and only `process.env` itself looks names up regardless of case: a copy of
 * it (`{ ...process.env }`) has no `PATH` there.
 */
export const pathOf = (env: Record<string, string | undefined>) => {
	if (env.PATH !== undefined || !windows()) return env.PATH ?? "";
	const key = Object.keys(env).find((name) => name.toUpperCase() === "PATH");
	return (key && env[key]) || "";
};

/** A regular file this process may run. Windows has no execute bit to check. */
export const isExecutable = async (file: string) => {
	try {
		if (!(await stat(file)).isFile()) return false;
		if (!windows()) await access(file, constants.X_OK);
		return true;
	} catch {
		return false;
	}
};

/** `file` with `.exe` where Windows names programs so and it has none. */
const native = (file: string) =>
	windows() && !/\.exe$/i.test(file) ? `${file}.exe` : file;

/**
 * Turns what a package manager's shim launches into the program itself:
 * given a script (`…/bin/codex.js`), the native binary it would start, if
 * there is one. A program has to be native to be run without a shell.
 */
export type ShimResolver = (target: string) => Promise<string | undefined>;

/**
 * What a Windows command shim (`name.cmd`, as npm, pnpm, yarn and fnm's
 * global installs write them) runs: the file it names beside itself, other
 * than the `node.exe` that runs a script.
 */
export const shimTarget = async (shim: string) => {
	let text: string;
	try {
		text = await readFile(shim, "utf8");
	} catch {
		return undefined;
	}
	const dir = dirname(shim);
	for (const [, relative] of text.matchAll(/"%~?dp0%?[\\/]([^"%]+)"/gi)) {
		if (!relative || /(^|[\\/])node(\.exe)?$/i.test(relative)) continue;
		return join(dir, ...relative.split(/[\\/]/));
	}
	return undefined;
};

/**
 * The native program `file` stands for: itself, or, for a Windows command
 * shim, the binary it launches (directly, or through `resolve` for a script).
 */
export const resolveProgram = async (
	file: string,
	resolve?: ShimResolver,
): Promise<string | undefined> => {
	if (windows() && /\.(cmd|bat)$/i.test(file)) {
		const target = await shimTarget(file);
		if (!target) return undefined;
		const program = /\.exe$/i.test(target) ? target : await resolve?.(target);
		return program && (await isExecutable(program)) ? program : undefined;
	}
	return (await isExecutable(file)) ? file : undefined;
};

/**
 * The first of a program on `env`'s `PATH`, then of `known` (written without
 * an extension: `.exe` is added on Windows), that can be run. On Windows a
 * directory without `name.exe` may hold the `name.cmd` shim of a global npm
 * install (fnm's, nvm's, npm's own), which is followed to its binary.
 */
export const findOnPath = async (
	name: string,
	env: Record<string, string | undefined>,
	known: string[] = [],
	resolve?: ShimResolver,
) => {
	const dirs = pathOf(env)
		.split(delimiter)
		.filter(Boolean)
		.map((dir) => dir.replace(/^"(.*)"$/, "$1"));
	const candidates = [
		...dirs.flatMap((dir) =>
			windows()
				? [join(dir, native(name)), join(dir, `${name}.cmd`)]
				: [join(dir, name)],
		),
		...known.map(native),
	];
	for (const candidate of candidates) {
		const program = await resolveProgram(candidate, resolve);
		if (program) return program;
	}
	return undefined;
};
