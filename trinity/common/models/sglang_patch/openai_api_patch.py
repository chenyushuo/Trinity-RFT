import time
from typing import Any, Dict, List, Optional

from fastapi import Request
from pydantic import model_serializer
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest as OriginalChatCompletionRequest,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionResponse as OriginalChatCompletionResponse,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionResponseChoice as OriginalChatCompletionResponseChoice,
)
from sglang.srt.entrypoints.openai.protocol import ChatMessage, SglExt
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat, logger
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.srt.entrypoints.openai.utils import (
    process_cached_tokens_details_from_ret,
    process_hidden_states_from_ret,
    process_routed_experts_from_ret,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.parser.reasoning_parser import ReasoningParser


# Add `return_token_ids` to the request
class ChatCompletionRequest(OriginalChatCompletionRequest):
    return_token_ids: bool = False


class ChatCompletionResponseChoice(OriginalChatCompletionResponseChoice):
    token_ids: Optional[List[int]] = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self.hidden_states is None:
            data.pop("hidden_states", None)
        if self.token_ids is None:
            data.pop("token_ids", None)
        return data


class ChatCompletionResponse(OriginalChatCompletionResponse):
    choices: List[ChatCompletionResponseChoice]
    prompt_token_ids: Optional[List[int]] = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self.prompt_token_ids is None:
            data.pop("prompt_token_ids", None)
        if self.sglext is None:
            data.pop("sglext", None)
        return data


class PatchedOpenAIServingChat(OpenAIServingChat):
    """This is a patched version of OpenAIServingChat which supports return
    `prompt_token_ids` and `token_ids` in non-streaming mode."""

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ):
        if not hasattr(request, "return_token_ids"):
            raise RuntimeError("You are using an unpatched version of OpenAIServingChat.")
        try:
            ret = await self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ).__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        if not isinstance(ret, list):
            ret = [ret]

        response = self._build_chat_response(
            request,
            adapted_request,
            ret,
            int(time.time()),
        )

        return response

    def _build_chat_response(
        self,
        request: ChatCompletionRequest,
        adapted_request: GenerateReqInput,
        ret: List[Dict[str, Any]],
        created: int,
    ):
        """Build chat completion response from generation results"""
        choices = []

        # Build sglext at response level (from first ret_item, as these are per-request)
        first_ret = ret[0]
        routed_experts = process_routed_experts_from_ret(first_ret, request)
        cached_tokens_details = process_cached_tokens_details_from_ret(first_ret, request)
        response_sglext = None
        if routed_experts or cached_tokens_details:
            response_sglext = SglExt(
                routed_experts=routed_experts,
                cached_tokens_details=cached_tokens_details,
            )

        for idx, ret_item in enumerate(ret):
            # Process logprobs
            choice_logprobs = None
            if request.logprobs:
                choice_logprobs = self._process_response_logprobs(ret_item)

            # Handle hidden states
            hidden_states = process_hidden_states_from_ret(ret_item, request)

            finish_reason = ret_item["meta_info"]["finish_reason"]
            text = ret_item["text"]

            # Handle reasoning content
            reasoning_text = None
            reasoning_parser = self.reasoning_parser
            if reasoning_parser and request.separate_reasoning:
                is_force_reasoning = (
                    self.template_manager.force_reasoning
                    or self._get_reasoning_from_request(request)
                )
                try:
                    parser = ReasoningParser(
                        model_type=reasoning_parser,
                        stream_reasoning=False,
                        force_reasoning=is_force_reasoning,
                        request=request,
                    )
                    reasoning_text, text = parser.parse_non_stream(text)
                except Exception as e:
                    logger.error(f"Reasoning parsing error: {e}")
                    return self.create_error_response(
                        "Failed to parse reasoning content",
                        err_type="InternalServerError",
                        status_code=500,
                    )

            # Handle tool calls
            tool_calls = None
            if request.tool_choice != "none" and request.tools and self.tool_call_parser:
                history_tool_calls_cnt = self._get_history_tool_calls_cnt(request)
                tool_calls, text, finish_reason = self._process_tool_calls(
                    text,
                    request.tools,
                    finish_reason,
                    request.tool_choice,
                    history_tool_calls_cnt,
                )

            choice_data = ChatCompletionResponseChoice(
                index=idx,
                message=ChatMessage(
                    role="assistant",
                    content=text if text else None,
                    tool_calls=tool_calls,
                    reasoning_content=reasoning_text if reasoning_text else None,
                ),
                logprobs=choice_logprobs,
                token_ids=(ret_item.get("output_ids") if request.return_token_ids else None),
                finish_reason=finish_reason["type"] if finish_reason else None,
                matched_stop=(
                    finish_reason["matched"]
                    if finish_reason and "matched" in finish_reason
                    else None
                ),
                hidden_states=hidden_states,
            )
            choices.append(choice_data)

        # Calculate usage
        usage = UsageProcessor.calculate_response_usage(
            ret,
            n_choices=request.n,
            enable_cache_report=self.tokenizer_manager.server_args.enable_cache_report,
        )

        response = ChatCompletionResponse(
            id=ret[0]["meta_info"]["id"],
            created=created,
            model=request.model,
            choices=choices,
            usage=usage,
            prompt_token_ids=(
                (
                    adapted_request.input_ids[0]
                    if adapted_request.input_ids and isinstance(adapted_request.input_ids[0], list)
                    else adapted_request.input_ids
                )
                if request.return_token_ids
                else None
            ),
            metadata={"weight_version": ret[0]["meta_info"]["weight_version"]},
            sglext=response_sglext,
        )
        return response

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ):
        if request.return_token_ids:
            raise ValueError("return_token_ids is not supported in streaming mode.")
        return await super()._handle_streaming_request(adapted_request, request, raw_request)
