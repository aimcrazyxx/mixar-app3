# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Provider-neutral request/response types used by the local agent path."""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class ProviderCapabilities:
    supports_tools: bool = True
    supports_parallel_tools: bool = True
    supports_vision: bool = True
    supports_streaming: bool = True
    supports_reasoning: bool = False
    supports_temperature: bool = True
    supports_top_p: bool = True
    supports_system_role: bool = True
    supports_responses_api: bool = False
    supports_chat_completions: bool = True
    supports_model_listing: bool = True


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class ProviderResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_content: str = ""
    continuation_items: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    endpoint: str = ""
    status_code: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    degraded_parameters: list[str] = field(default_factory=list)


class ProviderError(RuntimeError):
    """Safe error: never stores request headers, bodies, or API keys."""

    def __init__(self, message: str, *, status_code: int = 0, category: str = "provider"):
        super().__init__(message)
        self.status_code = int(status_code or 0)
        self.category = category


class AIProvider:
    capabilities: ProviderCapabilities

    def list_models(self) -> list[str]:
        raise NotImplementedError

    def test_connection(self) -> None:
        raise NotImplementedError

    def generate(
        self,
        messages: list[dict],
        *,
        tools: Optional[list[dict]] = None,
        on_text_delta: Optional[Callable[[str], None]] = None,
        cancel_event=None,
    ) -> ProviderResponse:
        raise NotImplementedError
