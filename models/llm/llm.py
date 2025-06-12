from typing import Mapping, Optional, Union, Generator
from urllib.parse import urljoin
import requests
from dify_plugin.entities.model import (
    AIModelEntity,
    DefaultParameterName,
    I18nObject,
    ModelFeature,
    ParameterRule,
    ParameterType,
)
from dify_plugin.entities.model.llm import (
    LLMMode,
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
)
from dify_plugin.entities.model.message import PromptMessage, PromptMessageTool

from dify_plugin.interfaces.model.openai_compatible.llm import (
    OAICompatLargeLanguageModel,
)

from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeError
)
import json
# 导入 logging 和自定义处理器
import logging
from dify_plugin.config.logger_format import plugin_logger_handler

# 使用自定义处理器设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(plugin_logger_handler)

class OpenAILargeLanguageModel(OAICompatLargeLanguageModel):
    def _update_endpoint_url(self, credentials: dict):
        credentials["extra_headers"] = {"X-Novita-Source": "dify.ai"}
        return credentials
    def get_customizable_model_schema(
        self, model: str, credentials: Mapping
    ) -> AIModelEntity:
        entity = super().get_customizable_model_schema(model, credentials)

        agent_though_support = credentials.get("agent_though_support", "not_supported")
        if agent_though_support == "supported":
            try:
                entity.features.index(ModelFeature.AGENT_THOUGHT)
            except ValueError:
                entity.features.append(ModelFeature.AGENT_THOUGHT)

        structured_output_support = credentials.get("structured_output_support", "not_supported")
        if structured_output_support == "supported":
            # ----
            # The following section should be added after the new version of `dify-plugin-sdks`
            # is released.
            # Related Commit:
            # https://github.com/langgenius/dify-plugin-sdks/commit/0690573a879caf43f92494bf411f45a1835d96f6
            # ----
            # try:
            #     entity.features.index(ModelFeature.STRUCTURED_OUTPUT)
            # except ValueError:
            #     entity.features.append(ModelFeature.STRUCTURED_OUTPUT)

            entity.parameter_rules.append(ParameterRule(
                name=DefaultParameterName.RESPONSE_FORMAT.value,
                label=I18nObject(en_US="Response Format", zh_Hans="回复格式"),
                help=I18nObject(
                    en_US="Specifying the format that the model must output.",
                    zh_Hans="指定模型必须输出的格式。",
                ),
                type=ParameterType.STRING,
                options=["text", "json_object", "json_schema"],
                required=False,
            ))
            entity.parameter_rules.append(ParameterRule(
                name=DefaultParameterName.JSON_SCHEMA.value,
                use_template=DefaultParameterName.JSON_SCHEMA.value
            ))

        if "display_name" in credentials and credentials["display_name"] != "":
            entity.label = I18nObject(
                en_US=credentials["display_name"], zh_Hans=credentials["display_name"]
            )

        entity.parameter_rules += [
            ParameterRule(
                name="enable_thinking",
                label=I18nObject(en_US="Thinking mode", zh_Hans="思考模式"),
                help=I18nObject(
                    en_US="Whether to enable thinking mode, applicable to various thinking mode models deployed on reasoning frameworks such as vLLM and SGLang, for example Qwen3.",
                    zh_Hans="是否开启思考模式，适用于vLLM和SGLang等推理框架部署的多种思考模式模型，例如Qwen3。",
                ),
                type=ParameterType.BOOLEAN,
                required=False,
            )
        ]
        return entity

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        enable_thinking = model_parameters.pop("enable_thinking", None)
        if enable_thinking is not None:
            model_parameters["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
        credentials["extra_headers"] = json.loads(credentials.pop("custom_headers", "{}"))
        return self._generate(
            model=model,
            credentials=credentials,
            prompt_messages=prompt_messages,
            model_parameters=model_parameters,
            tools=tools,
            stop=stop,
            stream=stream,
            user=user,
        )
    def _generate(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Invoke llm completion model

        :param model: model name
        :param credentials: credentials
        :param prompt_messages: prompt messages
        :param model_parameters: model parameters
        :param stop: stop words
        :param stream: is stream response
        :param user: unique user id
        :return: full response or stream response chunk generator result
        """
        headers = {
            "Content-Type": "application/json",
            "Accept-Charset": "utf-8",
        }
        extra_headers = credentials.get("extra_headers")
        if extra_headers is not None:
            headers = {
                **headers,
                **extra_headers,
            }

        api_key = credentials.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        endpoint_url = credentials["endpoint_url"]
        if not endpoint_url.endswith("/"):
            endpoint_url += "/"

        data = {"model": model, "stream": stream, **model_parameters}

        completion_type = LLMMode.value_of(credentials["mode"])

        if completion_type is LLMMode.CHAT:
            endpoint_url = urljoin(endpoint_url, "chat/completions")
            data["messages"] = [self._convert_prompt_message_to_dict(m, credentials) for m in prompt_messages]
        elif completion_type is LLMMode.COMPLETION:
            endpoint_url = urljoin(endpoint_url, "completions")
            data["prompt"] = prompt_messages[0].content
        else:
            raise ValueError("Unsupported completion type for model configuration.")

        # annotate tools with names, descriptions, etc.
        function_calling_type = credentials.get("function_calling_type", "no_call")
        formatted_tools = []
        if tools:
            if function_calling_type == "function_call":
                data["functions"] = [
                    {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
                    for tool in tools
                ]
            elif function_calling_type == "tool_call":
                data["tool_choice"] = "auto"

                for tool in tools:
                    formatted_tool = {
                        'type': 'function',
                        'function': {
                            'name': tool.name,
                            'description': tool.description,
                            'parameters': tool.parameters,
                        },
                    }
                    formatted_tools.append(formatted_tool)

                data["tools"] = formatted_tools

        if stop:
            data["stop"] = stop

        if user:
            data["user"] = user
        logger.info(f"headers: {headers}")
        response = requests.post(endpoint_url, headers=headers, json=data, timeout=(10, 300), stream=stream)

        if response.encoding is None or response.encoding == "ISO-8859-1":
            response.encoding = "utf-8"

        if response.status_code != 200:
            raise InvokeError(f"API request failed with status code {response.status_code}: {response.text}")

        if stream:
            return self._handle_generate_stream_response(model, credentials, response, prompt_messages)

        return self._handle_generate_response(model, credentials, response, prompt_messages)
    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials

        :param model: model name
        :param credentials: model credentials
        :return:
        """
        try:
            endpoint_url = credentials["endpoint_url"]
            if not endpoint_url.endswith("/"):
                endpoint_url += "/"
            endpoint_url = urljoin(endpoint_url, "completions")
            headers = json.loads(credentials.pop("custom_headers", "{}"))
            data = {"model": "deepseek-chat", "prompt": "<｜begin▁of▁sentence｜><｜User｜>你好<｜Assistant｜>", "max_tokens": 10}
            response = requests.post(endpoint_url, headers=headers, json=data, timeout=(10, 300), stream=False)

            if response.status_code != 200:
                raise CredentialsValidateFailedError(f"Api request failed with status code {response.status_code}: {response.text}")

        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))
