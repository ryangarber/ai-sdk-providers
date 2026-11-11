import { access, constants, stat } from "node:fs/promises";
import { delimiter, join } from "node:path";

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
 * The first of a program on `env`'s `PATH`, then of `known` (written without
 * an extension: `.exe` is added on Windows), that can be run.
 */
export const findOnPath = async (
	name: string,
	env: Record<string, string | undefined>,
	known: string[] = [],
) => {
	const candidates = [
		...pathOf(env)
			.split(delimiter)
			.filter(Boolean)
			.map((dir) => join(dir.replace(/^"(.*)"$/, "$1"), native(name))),
		...known.map(native),
	];
	for (const candidate of candidates)
		if (await isExecutable(candidate)) return candidate;
	return undefined;
};
