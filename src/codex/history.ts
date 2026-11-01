import type { LanguageModelV4ToolResultOutput } from "@ai-sdk/provider";
import type { CodexResponseItem } from "./host/app-server.ts";
import type { CodexHistoryEntry } from "./host/protocol.ts";
import { approvalCall, type MadeCall, type Tail, toolText } from "./prompt.ts";

const CALLS = new Set([
	"function_call",
	"custom_tool_call",
	"local_shell_call",
	"tool_search_call",
]);
const OUTPUTS = new Set([
	"function_call_output",
	"custom_tool_call_output",
	"local_shell_call_output",
	"tool_search_output",
]);

/** Codex's own context, which a thread adds again for itself. */
const isContext = (item: CodexResponseItem) => {
	if (item.type === "additional_tools") return true;
	if (item.type !== "message") return false;
	if (item.role === "developer" || item.role === "system") return true;
	if (item.role !== "user") return false;
	const text = item.content?.[0]?.text ?? "";
	return (
		/^<([a-z_]+)>[\s\S]*<\/\1>\s*$/.test(text.trim()) ||
		text.startsWith("# AGENTS.md instructions")
	);
};

/**
 * The model-visible history of a thread up to a step: through the response
 * it ended after, and the outputs of the calls that response made that came
 * before anything else. Undefined if the step can't be found, or the history
 * was compacted in its turn.
 */
export function sliceHistory(
	entries: CodexHistoryEntry[],
	cursor: { turnId: string; cursor: string },
): CodexResponseItem[] | undefined {
	let items: CodexResponseItem[] = [];
	let inTurn = false;
	let found = false;
	for (const entry of entries) {
		if (found) {
			// Only the outputs right after: the step's own tools.
			if (entry.type === "item" && OUTPUTS.has(entry.item.type)) {
				items.push(entry.item);
				continue;
			}
			break;
		}
		switch (entry.type) {
			case "turn":
				if (entry.turnId === cursor.turnId) inTurn = true;
				break;
			case "compacted":
				if (inTurn || !entry.items) return undefined;
				items = [...entry.items];
				break;
			case "item":
				if (entry.turnId === cursor.turnId) inTurn = true;
				items.push(entry.item);
				break;
			case "response":
				if (entry.responseId === cursor.cursor) found = true;
				break;
		}
	}
	return found ? items : undefined;
}

/** A tool's answer the thread never saw, as what to tell the model of it. */
export interface DanglingAnswer {
	call: MadeCall;
	result?: LanguageModelV4ToolResultOutput;
	approval?: { approved: boolean; reason?: string };
}

/** What the caller gave the calls left waiting at the cursor. */
export function answersFor(dangling: MadeCall[], tail: Tail): DanglingAnswer[] {
	return dangling.map((call) => {
		const result = tail.results.find((r) => r.toolCallId === call.id);
		const approval = tail.approvals.find(
			(a) => approvalCall(a.approvalId) === call.id,
		);
		return {
			call,
			...(result && { result: result.output }),
			...(approval && {
				approval: { approved: approval.approved, reason: approval.reason },
			}),
		};
	});
}

const describeAnswer = ({ call, result, approval }: DanglingAnswer) => {
	if (result) return `${call.name}: ${toolText(result)}`;
	if (approval?.approved)
		return `${call.name}: approved, but it didn't run before the session restarted. Make the call again if it is still needed.`;
	if (approval)
		return `${call.name}: the user denied it${approval.reason ? `: ${approval.reason}` : "."}`;
	return `${call.name}: it didn't run.`;
};

/**
 * Items for a thread rebuilt at a step: its history without Codex's own
 * context, and an output for each call it left waiting, from what the caller
 * answered.
 */
export function rebuildItems(
	history: CodexResponseItem[],
	answers: DanglingAnswer[],
	options: { keepReasoning: boolean },
): CodexResponseItem[] {
	const items: CodexResponseItem[] = [];
	const open = new Map<string, CodexResponseItem>();
	for (const item of history) {
		if (isContext(item)) continue;
		if (item.type === "reasoning" && !options.keepReasoning) continue;
		const { internal_chat_message_metadata_passthrough: _, ...rest } = item;
		items.push(rest as CodexResponseItem);
		if (CALLS.has(item.type) && item.call_id)
			open.set(item.call_id, rest as CodexResponseItem);
		if (OUTPUTS.has(item.type) && item.call_id) open.delete(item.call_id);
	}

	// Calls the caller answered directly, by their call id.
	const direct = new Map(answers.map((answer) => [answer.call.id, answer]));
	const nested = answers.filter((answer) => !open.has(answer.call.id));
	const scripts = [...open.values()].filter(
		(item) => item.type === "custom_tool_call",
	);
	const lastScript = scripts.at(-1);
	for (const [callId, item] of open) {
		if (item.type === "tool_search_call") {
			items.splice(items.indexOf(item), 1);
			continue;
		}
		let text: string;
		const answer = direct.get(callId);
		if (answer?.result && item.name === "request_user_input")
			text = toolText(answer.result);
		else if (answer) text = describeAnswer(answer);
		else if (item === lastScript && nested.length)
			text = [
				"The script stopped when the session restarted, so it didn't finish. What its calls got:",
				...nested.map(describeAnswer),
			].join("\n");
		else text = "This call stopped when the session restarted.";
		items.push(
			item.type === "custom_tool_call"
				? { type: "custom_tool_call_output", call_id: callId, output: text }
				: item.type === "local_shell_call"
					? { type: "local_shell_call_output", call_id: callId, output: text }
					: { type: "function_call_output", call_id: callId, output: text },
		);
	}
	return items;
}
