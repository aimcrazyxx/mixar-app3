# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Provider-neutral iterative Mixar agent loop."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from .provider_types import ProviderError

SYSTEM_PROMPT = """You are the Mixar AI Agent, an autonomous 3D creation and Blender assistant integrated directly into Mixar.

Your job is not merely to tell the user how to perform tasks. When the execute_blender_python tool is available, use it to inspect, create, edit, fix, render, generate, import, organize, and validate the user's project.

Before making assumptions, inspect the current project state. For complex tasks: understand the request, inspect relevant state, plan internally, execute with tools, inspect the result, correct problems, and continue until complete. Do not stop after one tool call if more work is necessary. Treat every tool result as new information. If a tool fails, analyze the error and attempt a reasonable correction. Never claim success unless tool output or a follow-up inspection confirms it. Preserve existing user work unless the request requires changing it. Prefer executing over describing manual steps when execution tools are available.

The tool runs sandboxed Python with bpy, bmesh, mathutils, json, math, random, re, and numpy already available. Set __RESULT__ to a concise JSON-serializable dict when inspection data should be returned. Use multiple tool calls when independent actions can safely run in parallel, and always perform a final inspection for scene-changing tasks."""

BLENDER_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_blender_python",
        "description": "Execute sandboxed Python on Blender's main thread. Use for inspection and for actual scene/project changes. Return structured data by assigning __RESULT__.",
        "parameters": {
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": "Python source using bpy. Keep each call focused and inspect after mutations.",
                }
            },
            "required": ["script"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class AgentRunResult:
    text: str
    messages: list[dict]
    debug: dict


def approximate_tokens(messages: list[dict], tools: list[dict]) -> int:
    try:
        raw = json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False)
    except TypeError:
        raw = str(messages) + str(tools)
    return max(1, len(raw) // 4)


def trim_context(messages: list[dict], limit: int) -> list[dict]:
    """Preserve system/current request and never split tool-call groups."""
    if limit <= 0 or approximate_tokens(messages, []) <= limit:
        return list(messages)
    system = [m for m in messages if m.get("role") == "system"][:1]
    tail = [m for m in messages if m.get("role") != "system"]
    groups = []
    for message in tail:
        if message.get("role") == "tool" and groups:
            groups[-1].append(message)
        else:
            groups.append([message])
    current_group = max(
        (
            index
            for index, group in enumerate(groups)
            if any(message.get("role") == "user" for message in group)
        ),
        default=max(0, len(groups) - 1),
    )
    # Keep the current user request and every assistant/tool exchange after it,
    # even when that required set itself exceeds the configured soft limit.
    kept_groups = groups[current_group:]
    for group in reversed(groups[:current_group]):
        candidate_groups = [group, *kept_groups]
        candidate = system + [item for block in candidate_groups for item in block]
        if approximate_tokens(candidate, []) > limit:
            # Keep one contiguous recent suffix. Skipping this group and then
            # accepting an older, smaller one would reorder the conversation's
            # effective meaning and could separate a request from its answer.
            break
        kept_groups.insert(0, group)
    return system + [item for block in kept_groups for item in block]


def _assistant_wire(response) -> dict:
    message = {"role": "assistant", "content": response.content or ""}
    if response.reasoning_content:
        message["reasoning_content"] = response.reasoning_content
    if response.continuation_items:
        message["_mixar_responses_output"] = response.continuation_items
    if response.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in response.tool_calls
        ]
    return message


def run_agent_loop(
    provider,
    messages: list[dict],
    execute_tool,
    *,
    max_iterations: int = 20,
    context_limit: int = 0,
    cancel_event=None,
    on_text_delta=None,
) -> AgentRunResult:
    tools = [BLENDER_TOOL] if provider.capabilities.supports_tools else []
    transcript = trim_context(messages, context_limit)
    debug = {
        "provider": "OpenAI Compatible",
        "model": provider.config.model,
        "base_url": provider.config.base_url,
        "endpoint": "",
        "context_tokens_approx": approximate_tokens(transcript, tools),
        "message_count": len(transcript),
        "tool_count": len(tools),
        "tool_calls": [],
        "tool_results": [],
        "iterations": 0,
        "duration_ms": 0,
        "http_status": 0,
        "finish_reason": "",
        "degraded_parameters": [],
        "capability_warnings": [],
        "usage": [],
        "reasoning_present": False,
        "request_count": 0,
    }
    started = time.monotonic()
    final_text = ""
    try:
        for iteration in range(1, max(1, max_iterations) + 1):
            if cancel_event is not None and cancel_event.is_set():
                raise ProviderError("Agent run cancelled", category="cancelled")
            debug["iterations"] = iteration
            response = provider.generate(
                transcript,
                tools=tools or None,
                on_text_delta=on_text_delta,
                cancel_event=cancel_event,
            )
            degraded = list(debug["degraded_parameters"])
            for parameter in response.degraded_parameters:
                if parameter not in degraded:
                    degraded.append(parameter)
            warnings = list(debug["capability_warnings"])
            warning_map = {
                "tools": "The provider rejected tool calling; Blender actions could not be executed.",
                "reasoning": "The provider rejected the configured reasoning mode.",
                "reasoning_effort": "The provider rejected the configured reasoning effort.",
            }
            for parameter in response.degraded_parameters:
                warning = warning_map.get(parameter)
                if warning and warning not in warnings:
                    warnings.append(warning)
            debug.update(
                {
                    "endpoint": response.endpoint,
                    "http_status": response.status_code,
                    "finish_reason": response.finish_reason,
                    "degraded_parameters": degraded,
                    "capability_warnings": warnings,
                    "reasoning_present": bool(
                        debug["reasoning_present"] or response.reasoning_content
                    ),
                    "request_count": debug["request_count"] + 1,
                }
            )
            if response.usage:
                debug["usage"].append(
                    {"iteration": iteration, "values": response.usage}
                )
            for call_index, call in enumerate(response.tool_calls):
                if not call.id:
                    call.id = f"call_{iteration}_{call_index}"
            transcript.append(_assistant_wire(response))
            if not response.tool_calls:
                final_text = response.content.strip()
                if not final_text:
                    final_text = "The provider returned an empty response."
                if warnings:
                    final_text += "\n\nProvider capability notice: " + " ".join(
                        warnings
                    )
                break

            for call in response.tool_calls:
                debug["tool_calls"].append({"iteration": iteration, "name": call.name})
                if cancel_event is not None and cancel_event.is_set():
                    raise ProviderError("Agent run cancelled", category="cancelled")
                try:
                    args = json.loads(call.arguments or "{}")
                    if call.name != "execute_blender_python":
                        result = {
                            "success": False,
                            "error": f"Unknown tool: {call.name}",
                        }
                    elif not isinstance(args, dict) or not isinstance(
                        args.get("script"), str
                    ):
                        result = {
                            "success": False,
                            "error": "Tool argument 'script' must be a string",
                        }
                    else:
                        result = execute_tool(call.name, args)
                except json.JSONDecodeError as exc:
                    result = {
                        "success": False,
                        "error": f"Invalid tool JSON: {exc.msg}",
                    }
                except Exception as exc:
                    result = {
                        "success": False,
                        "error": f"Tool execution failed: {type(exc).__name__}: {exc}",
                    }
                debug["tool_results"].append(
                    {
                        "iteration": iteration,
                        "name": call.name,
                        "success": bool(result.get("success"))
                        if isinstance(result, dict)
                        else False,
                    }
                )
                transcript.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )
        else:
            final_text = f"Stopped after {max_iterations} agent iterations before a final answer."
    finally:
        debug["duration_ms"] = round((time.monotonic() - started) * 1000)
        debug["message_count"] = len(transcript)
    return AgentRunResult(final_text, transcript, debug)
