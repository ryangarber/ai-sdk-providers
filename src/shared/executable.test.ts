import { chmod, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { findOnPath, pathOf, shimTarget } from "./executable.ts";

const platform = process.platform;
const setPlatform = (value: NodeJS.Platform) =>
	Object.defineProperty(process, "platform", { value });

afterEach(() => setPlatform(platform));

describe("pathOf", () => {
	it("reads Windows' `Path` from a copy of the environment", () => {
		setPlatform("win32");
		expect(pathOf({ Path: "C:\\bin" })).toBe("C:\\bin");
		expect(pathOf({ PATH: "C:\\a", Path: "C:\\b" })).toBe("C:\\a");
	});

	it("is case-sensitive elsewhere", () => {
		setPlatform("linux");
		expect(pathOf({ Path: "/bin" })).toBe("");
	});
});

describe("findOnPath", () => {
	it("finds a program on PATH, then in the known places", async () => {
		// As named elsewhere: Windows' would be `tool.exe`.
		setPlatform("linux");
		const dir = await mkdtemp(join(tmpdir(), "executable-"));
		try {
			const bin = join(dir, "bin");
			await mkdir(bin);
			const file = join(bin, "tool");
			await writeFile(file, "");
			await chmod(file, 0o755);
			expect(
				await findOnPath("tool", { PATH: `${dir}${delimiter}${bin}` }),
			).toBe(file);
			expect(await findOnPath("tool", { PATH: "" }, [file])).toBe(file);
			// A directory of that name is not the program.
			expect(await findOnPath("bin", { PATH: dir })).toBeUndefined();
		} finally {
			await rm(dir, { recursive: true, force: true });
		}
	});
});

describe("Windows command shims", () => {
	const NPM = String.raw`@ECHO off
SETLOCAL
IF EXIST "%dp0%\node.exe" (
  SET "_prog=%dp0%\node.exe"
) ELSE (
  SET "_prog=node"
)
endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & "%_prog%"  "%dp0%\node_modules\@openai\codex\bin\codex.js" %*
`;
	const NPM_NATIVE = String.raw`@ECHO off
"%dp0%\node_modules\@anthropic-ai\claude-code\bin\claude.exe"   %*
`;
	const PNPM = String.raw`@SETLOCAL
@IF EXIST "%~dp0\node.exe" (
  "%~dp0\node.exe"  "%~dp0\node_modules\corepack\dist\pnpm.js" %*
) ELSE (
  node  "%~dp0\node_modules\corepack\dist\pnpm.js" %*
)
`;

	it("reads what npm's and pnpm's shims launch, past node", async () => {
		const dir = await mkdtemp(join(tmpdir(), "shim-"));
		try {
			const write = async (name: string, text: string) => {
				await writeFile(join(dir, name), text);
				return join(dir, name);
			};
			expect(await shimTarget(await write("codex.cmd", NPM))).toBe(
				join(dir, "node_modules", "@openai", "codex", "bin", "codex.js"),
			);
			expect(await shimTarget(await write("claude.cmd", NPM_NATIVE))).toBe(
				join(
					dir,
					"node_modules",
					"@anthropic-ai",
					"claude-code",
					"bin",
					"claude.exe",
				),
			);
			expect(await shimTarget(await write("pnpm.cmd", PNPM))).toBe(
				join(dir, "node_modules", "corepack", "dist", "pnpm.js"),
			);
		} finally {
			await rm(dir, { recursive: true, force: true });
		}
	});

	it("finds a native program behind a shim on PATH", async () => {
		setPlatform("win32");
		const dir = await mkdtemp(join(tmpdir(), "shim-"));
		try {
			const bin = join(dir, "node_modules", "tool", "bin");
			await mkdir(bin, { recursive: true });
			const exe = join(bin, "tool.exe");
			const script = join(bin, "tool.js");
			await writeFile(exe, "");
			await writeFile(script, "");
			await writeFile(
				join(dir, "tool.cmd"),
				String.raw`"%dp0%\node_modules\tool\bin\tool.exe" %*`,
			);
			expect(await findOnPath("tool", { Path: dir })).toBe(exe);

			// A script is followed only as far as the resolver takes it.
			await writeFile(
				join(dir, "tool.cmd"),
				String.raw`"%dp0%\node.exe" "%dp0%\node_modules\tool\bin\tool.js" %*`,
			);
			expect(await findOnPath("tool", { Path: dir })).toBeUndefined();
			expect(
				await findOnPath("tool", { Path: dir }, [], async (target) =>
					target === script ? exe : undefined,
				),
			).toBe(exe);
		} finally {
			await rm(dir, { recursive: true, force: true });
		}
	});
});
