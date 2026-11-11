import { chmod, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { findOnPath, pathOf } from "./executable.ts";

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
