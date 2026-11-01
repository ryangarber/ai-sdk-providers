import type { AfmizeClient, AfmModel } from "@afmize/node";
import { z } from "zod";

/** A model afmize serves. */
export type AfmModelId = AfmModel;

/** `providerOptions.afm`. */
export const AfmOptions = z.object({
	/**
	 * How much the model reasons (macOS 27 / iOS 27); overrides the call's
	 * `reasoning`.
	 */
	reasoningLevel: z.enum(["light", "moderate", "deep"]).optional(),
});
export type AfmOptions = z.infer<typeof AfmOptions>;

export interface AfmProviderSettings {
	/**
	 * How to reach afmize: a client, or a function that makes one on first
	 * use. Defaults to `@afmize/node`'s, which loads afmize's prebuilt library
	 * under Bun or Node on macOS. In a Capacitor app on iOS, pass
	 * `@afmize/capacitor`'s `createAfmize()`; anything else implementing
	 * {@link AfmizeClient} works too, e.g. a webview's over its host's IPC.
	 */
	afmize?: AfmizeClient | (() => AfmizeClient | Promise<AfmizeClient>);
}
