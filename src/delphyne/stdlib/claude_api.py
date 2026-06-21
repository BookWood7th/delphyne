import json
from delphyne.stdlib import models as md
from delphyne.utils.yaml import pretty_yaml
from typing import Any, override
from dataclasses import dataclass, replace

from collections.abc import Sequence, AsyncIterable
import anthropic
from anthropic import types as anthropicTypes
from anthropic._types import omit, Omit
from anthropic import Anthropic
from delphyne.stdlib.openai_api import ToolCallIdGenerator
from delphyne.core.streams import Budget
from delphyne.core import ToolCall
from delphyne.core.refs import Structured

def translate_chat(
    chat: md.Chat,
) -> Sequence[anthropicTypes.MessageParam]:
    """
    We translate the chat into the format expected by OpenAI API.

    Unique ids are generated for tool calls.
    """
    gen = ToolCallIdGenerator()

    def translate(msg: md.ChatMessage) -> anthropicTypes.MessageParam:
        match msg:
            case md.SystemMessage(content=content):
                return {"role": "system", "content": content}
            case md.UserMessage(content=content):
                return {"role": "user", "content": content}
            case md.AssistantMessage(answer=answer):
                tool_use_block = None
                if answer.tool_calls:
                    tool_use_block = [
                        {
                            "type": "tool_use",
                            "id": gen.get_id(call),
                            "name": call.name,
                            "input": json.dumps(call.args, indent=2)
                        }
                        for call in answer.tool_calls
                    ]

                if isinstance(answer.content, str):
                    content = [
                        {
                            "type": "text",
                            "text": answer.content
                        }                            
                    ]
                    if tool_use_block:
                        for tool_use in tool_use_block:
                            content.append(tool_use)
                    content = json.dumps(content, indent=2)
                else:
                    # We serialize the structured answer
                    content = json.dumps(answer.content.structured, indent=2)
                
                if answer.justification is not None:
                #Not supported by Claude
                    pass
                #    content += f"\n\n{answer.justification}"
                res: anthropicTypes.MessageParam = {
                    "role": "assistant",
                    "content": content,
                }
                return res
            case md.ToolMessage(call=call, result=result):
                if isinstance(result, str):
                    content = result
                else:
                    content = pretty_yaml(result.structured)
                return {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": gen.get_id(call),
                            "content": content
                        }
                    ]
                }

    return [translate(msg) for msg in chat]


def _patch_prefix_items_arrays(schema: Any) -> Any:
    """
    Recursively patch a JSON schema *in place* so that every array with
    `prefixItems` also has `"items": false"`. Such arrays are used to
    represent tuples. This fixes OpenAI's "array schema missing items"
    error.
    """
    if isinstance(schema, dict):
        if (
            schema.get("type") == "array"  # type: ignore
            and "prefixItems" in schema
            and "items" not in schema
        ):
            schema["items"] = {"type": "null"}
        for v in schema.values():  # type: ignore
            _patch_prefix_items_arrays(v)
    elif isinstance(schema, list):
        for v in schema:  # type: ignore
            _patch_prefix_items_arrays(v)
    return schema  # type: ignore


def _strict_schema(schema: Any):
    from copy import deepcopy

    from openai.lib._pydantic import _ensure_strict_json_schema  # type: ignore

    schema = deepcopy(schema)
    _ensure_strict_json_schema(schema, path=(), root=schema)
    _patch_prefix_items_arrays(schema)
    return schema

def _base_budget(n: int, model_class: str | None = None) -> dict[str, float]:
    budget: dict[str, float] = {md.NUM_REQUESTS: 1, md.NUM_COMPLETIONS: n}
    if model_class is not None:
        budget[md.budget_entry("num_requests", model_class)] = 1
        budget[md.budget_entry("num_completions", model_class)] = 1

    return budget

def _make_chat_tool(tool: md.Schema) -> anthropicTypes.ToolUnionParam:
    ret: anthropicTypes.ToolUnionParam = {
        "name": tool.name,
        "input_schema": _strict_schema(tool.schema),
    }

    if tool.description is not None:
        ret["description"] = tool.description

    return ret

def _compute_spent_budget(
    n: int,
    model_class: str | None = None,
    pricing: md.ModelPricing | None = None,
    usage: anthropic.types.Usage | None = None,
) -> dict[str, float]:

    budget = _base_budget(n, model_class)

    def add(cat: md.BudgetCategory, value: float):
        budget[md.budget_entry(cat)] = value
        if model_class is not None:
            budget[md.budget_entry(cat, model_class)] = value

    if usage is not None:
        # Claude equivalents
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens

        add("input_tokens", input_tokens)
        add("output_tokens", output_tokens)

        if pricing is not None:
            non_cached = input_tokens  # everything is non-cached in Claude
            price = (
                pricing.dollars_per_input_token * non_cached
                + pricing.dollars_per_output_token * output_tokens
            )
            add("price", price)

    return budget

def _convert_tool_choice(value: str | None) -> anthropicTypes.ToolChoiceParam | Omit:
    if value == "auto":
        return {"type": "auto"}
    if value == "required":
        return {"type": "any"}
    if value == "none":
        return omit
    return omit


@dataclass(kw_only=True)
class ClaudeCompatibleModel(md.LLM):
    """
    A Model accessible via an OpenAI-compatible API.

    Attributes:

        options: the default options to use for requests.
        api_key: the API key to use for authentication.
        base_url: the base URL of the OpenAI-compatible API.
        model_class: an optional identifier for the model class (e.g.,
            "reasoning_large"). When provided, class-specific budget
            metrics are reported, so that resource consumption can be
            tracked separately for different classes of models (e.g.,
            tracking "num_requests__reasoning_large" separately from
            "num_requests__chat_small").
        pricing: pricing information for the model.
        no_json_schema: if `True`, JSON mode is used for structured
            output instead of JSON Schema. This is useful for providers
            like DeepSeek that do not support structured output with
            schemas.
    """

    options: md.RequestOptions
    api_key: str | None = None
    base_url: str | None = None
    no_json_schema: bool = False
    model_class: str | None = None
    pricing: md.ModelPricing | None = None

    @override
    def add_model_defaults(self, req: md.LLMRequest) -> md.LLMRequest:
        return replace(req, options=self.options | req.options)

    @override
    def estimate_budget(self, req: md.LLMRequest) -> Budget:
        return Budget(_base_budget(req.num_completions, self.model_class))

    @override
    def _send_final_request(self, req: md.LLMRequest) -> md.LLMResponse:
        client = Anthropic(api_key=self.api_key)
        options = req.options
        assert "model" in options, "No model was specified"
        tools = [_make_chat_tool(tool) for tool in req.tools]
        

        
        tool_choice = _convert_tool_choice(options.get("tool_choice", None))

        try:
            anthropicTypes.ToolChoiceParam
            options.get("tool_choice")
            response: anthropicTypes.message.Message = client.messages.create(
                model=options["model"],
                messages=translate_chat(req.chat),
                max_tokens=options.get("max_completion_tokens", 0),
                temperature=options.get("temperature", omit),
                tools=tools if tools else omit,
                tool_choice=tool_choice
            )
        except (anthropic.RateLimitError, anthropic.APITimeoutError) as e:
            raise md.LLMBusyException(e)
        outputs: list[md.LLMOutput] = []
        log: list[md.LLMResponseLogItem] = []

        content_blocks = response.content
        stop_reason = response.stop_reason

        content_text = ""
        tool_calls: list[ToolCall] = []
        has_tool_use = False

        # ----------------------------
        # 1. Parse content blocks
        # ----------------------------
        for block in content_blocks:
            if block.type == "text":
                content_text += block.text

            elif block.type == "tool_use":
                has_tool_use = True
                try:
                    tool_calls.append(
                        ToolCall(block.name, block.input)
                    )
                except Exception:
                    log.append(
                        md.LLMResponseLogItem(
                            "error",
                            "failed_to_parse_tool_call",
                            metadata={"tool_call": block},
                        )
                    )
                    continue

        # ----------------------------
        # 2. Finish reason mapping
        # ----------------------------
        def map_finish_reason(stop_reason: str | None) -> md.FinishReason:
            if stop_reason == "tool_use":
                return "tool_calls"

            if stop_reason == "max_tokens":
                return "length"

            if stop_reason in ("end_turn", "stop_sequence", None):
                return "stop"

            # Claude doesn't have content_filter equivalent
            return "stop"
        
        finish_reason: md.FinishReason = map_finish_reason(stop_reason)

        # ----------------------------
        # 3. Empty / refusal handling
        # ----------------------------

        budget = Budget(
            _compute_spent_budget(
                req.num_completions,
                self.model_class,
                self.pricing,
                response.usage,
            )
        )
        if not content_text and not has_tool_use:
            log.append(md.LLMResponseLogItem("error", "empty_answer"))
            return md.LLMResponse(outputs, budget, log, response.model, None)

        # Claude does not expose structured refusal,
        # so treat empty + end_turn as failure signal if needed

        # ----------------------------
        # 4. Structured output handling
        # ----------------------------

        if (
            req.structured_output is not None
            and not has_tool_use
        ):
            try:
                content = Structured(json.loads(content_text))
            except Exception as e:
                log.append(
                    md.LLMResponseLogItem(
                        "error",
                        "failed_to_parse_structured_output",
                        metadata={"content": content_text, "error": str(e)},
                    )
                )
                return md.LLMResponse(outputs, budget, log, response.model, None)
        else:
            content = content_text

        # ----------------------------
        # 5. Logprobs (NOT SUPPORTED IN CLAUDE)
        # ----------------------------
        logprobs = None  # removed entirely

        # ----------------------------
        # 6. Reasoning content (optional / non-standard)
        # ----------------------------
        reasoning_content = None  # Claude does not provide this field

        # ----------------------------
        # 7. Output object
        # ----------------------------
        output = md.LLMOutput(
            content=content,
            logprobs=logprobs,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )

        outputs.append(output)

        # ----------------------------
        # 8. Usage
        # ----------------------------
        usage = response.usage.model_dump() if response.usage else None

        budget = Budget(
            _compute_spent_budget(
                req.num_completions,
                self.model_class,
                self.pricing,
                response.usage,
            )
        )

        return md.LLMResponse(outputs, budget, log, response.model, usage)

    async def stream_request(
        self, chat: md.Chat, options: md.RequestOptions
    ) -> AsyncIterable[str]:

        client = anthropic.AsyncAnthropic(api_key=self.api_key, base_url=self.base_url)

        options = self.options | options
        assert "model" in options, "No model was specified"

        async with client.messages.stream(
            model=options["model"],
            messages=translate_chat(chat),
            max_tokens=options.get("max_completion_tokens", 1024),
            temperature=options.get("temperature", omit),
            tools=options.get("tools", omit),
            tool_choice=_convert_tool_choice(options.get("tool_choice", None)),
        ) as stream:

            async for event in stream:
                # Text delta events (this replaces OpenAI delta.content)
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield event.delta.text


