import { createLocalHost } from "./host/local.ts";
import { type CursorProvider, createCursor } from "./provider.ts";

export * from "./host/local.ts";

/** Cursor in this process, on `CURSOR_API_KEY` or a saved login. */
export const cursor: CursorProvider = createCursor({
	host: createLocalHost(),
});
