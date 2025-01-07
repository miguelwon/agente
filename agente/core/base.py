import asyncio
import json
import traceback
import warnings
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from litellm import acompletion
from pydantic import BaseModel, Field

from function_schema import Doc, get_function_schema

from ..models.schemas import (
    Message, Response, StreamResponse, ConversationHistory,
    ToolCall, FunctionCall,Usage
)
from .decorators import function_tool



class BaseAgent(BaseModel):
    """
    A base class for AI agents with tool execution capabilities.
    """

    agent_name: str
    is_conversational: bool = True
    parent_agent: Optional["BaseAgent"] = None
    child_agents: List["BaseAgent"] = []
    conv_history: ConversationHistory = Field(
        default_factory=lambda: ConversationHistory(messages=[])
    )
    tools_mem: List[Dict[str, Any]] = []
    logs_completions: List[Any] = []
    task_complete: bool = False

    responses: List[Response] = []
    stream_responses: List[StreamResponse] = []
    agents_queue: Optional[Deque["BaseAgent"]] = Field(None, exclude=True)

    # These will be populated during model_post_init
    tools: List[Callable] = Field([], exclude=True)
    tools_schema: List[Dict[str, Any]] = Field([], exclude=True)
    tools_functions: Dict[str, Callable] = Field({}, exclude=True)
    tools_agent: Dict[str, Callable] = Field({}, exclude=True)


    completion_kwargs: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "arbitrary_types_allowed": True
    }

    @property
    def completion_config(self) -> Dict[str, Any]:
        """Merged completion configuration with defaults and overrides."""
        return self.completion_kwargs

    def model_post_init(self, __context) -> None:
        """Initialize after Pydantic validation"""

        default_kwargs = {"model": "gpt-4o", "stream": False}
        self.completion_kwargs = {**default_kwargs, **self.completion_kwargs}

        self.tools, self.tools_schema = self._discover_tools()

        if self.agents_queue is None:
            self.agents_queue = deque()

        if not self.parent_agent:
            self.agents_queue.appendleft(self)

    def _discover_tools(self) -> Tuple[List[Callable], List[Dict[str, Any]]]:
        """
        Discover and register tools (methods) marked with the @tool decorator.
        """
        tools = []
        schemas = []
        seen_names = set()

        for cls in reversed(self.__class__.__mro__):
            for name, method in vars(cls).items():
                if (
                    name == "complete_task"
                    and cls is BaseTaskAgent
                    and self.__class__ is not BaseTaskAgent
                ):
                    continue

                if name in seen_names:
                    continue

                if callable(method) and getattr(method, "is_tool", False):
                    tools.append(method)
                    schema = get_function_schema(method)

                    if (
                        "parameters" in schema
                        and "properties" in schema["parameters"]
                    ):
                        schema["parameters"]["properties"].pop("self", None)
                        if "required" in schema["parameters"]:
                            schema["parameters"]["required"] = [
                                param
                                for param in schema["parameters"]["required"]
                                if param != "self"
                            ]
                    schemas.append({"type": "function", "function": schema})

                    seen_names.add(name)

        self.tools_functions = {
            tool.__name__: tool
            for tool in tools
            if not getattr(tool, "is_agent", False)
        }
        self.tools_agent = {
            tool.__name__: tool for tool in tools if getattr(tool, "is_agent", False)
        }

        return tools, schemas

    async def run(self,max_retries:int = 5,n_call:int = 1) -> Any:
        """
        Run the agent asynchronously, processing messages and executing tools.
        """
        agent = self.agents_queue[0]

        n_call += 1
        if n_call > max_retries:
            warnings.warn(f"Max retries ({max_retries}) reached for agent {self.agent_name}")
            self.add_message("assistant","Max retries reached")
            return



        if agent.tools_mem:
            raise ValueError("There are still tools in memory to be executed")

        messages = [
            message.model_dump(exclude_unset=True, exclude={"agent_name", "hidden", "id"})
            for message in agent.conv_history.messages
            if message.agent_name == agent.agent_name
        ]

        completion_params = {
            **agent.completion_config,
            "messages": messages,
        }

        if agent.tools_schema:
            completion_params.update({
                "tools": agent.tools_schema,
                "tool_choice": "auto"
            })

        if completion_params["stream"]:
            async for item in agent._run_stream(completion_params,max_retries,n_call):
                yield item
        else:
            async for item in agent._run_no_stream(completion_params, max_retries,n_call):
                yield item

    async def _run_no_stream(
        self,
        completion_params: Dict,
        max_retries,
        n_call
    ):
        """
        Run the agent asynchronously without streaming the response.
        """

        

        _response = await acompletion(**completion_params)
        self.logs_completions.append(_response)

        content = _response.choices[0].message.content
        tool_calls = []
        if _response.choices[0].message.tool_calls:
            tool_calls = [
                ToolCall(
                    index=tool_call.index if "index" in tool_call else i,
                    id=tool_call.id,
                    function=FunctionCall(
                        arguments=tool_call.function.arguments,
                        name=tool_call.function.name,
                    ),
                    type="function",
                )
                for i, tool_call in enumerate(_response.choices[0].message.tool_calls)
            ]

        if hasattr(_response,"usage"):
            usage = Usage(completion_tokens=_response.usage.completion_tokens,
                          prompt_tokens=_response.usage.prompt_tokens,
                          total_tokens=_response.usage.total_tokens)

        response = Response(
            call_id=_response.id,
            agent_name=self.agent_name,
            role=_response.choices[0].message.role,
            content=content or "",
            tool_calls=[t.model_dump() for t in tool_calls],
            usage=usage if hasattr(_response,"usage") else None
        )
        self.responses.append(response)

        yield response

        if content:
            self._add_message(role="assistant", content=content)
        if tool_calls:
            complete_task = False
            for tool in tool_calls:
                if tool.function.name == "complete_task":
                    complete_task = True
                self.tools_mem.append(tool.model_dump())
            self._add_message(role="assistant", tool_calls=self.tools_mem)

            #first we execute the function tools    
            await self._execute_function_tools()

            #then we enqueue the agent tools
            self._enqueue_agent_tools()


            if complete_task:
                self.task_complete = True
                self.agents_queue.remove(self)

            #now the same but for current agent (or call again after the tools were executed) 
            async for response in self.run(max_retries,n_call):
                yield response

    async def _run_stream(
        self,
        completion_params: Dict,
        max_retries,
        n_call
    ):
        """
        Run the agent asynchronously with streaming the response.
        """
        tool_id = None
        tool_name = None
        role = "assistant"
        tool_calls_info = defaultdict(lambda: defaultdict(str))
        current_content = ""

        async for chunk in await acompletion(**completion_params):
            self.logs_completions.append(chunk)

            delta = chunk.choices[0].delta
            content = None

            if hasattr(delta, "content") and delta.content:
                current_content += delta.content
                content = delta.content

            elif hasattr(delta, "tool_calls") and delta.tool_calls:
                if delta.tool_calls[0].id is not None:
                    tool_id = delta.tool_calls[0].id
                    tool_name = delta.tool_calls[0].function.name
                    tool_calls_info[tool_id]["name"] = tool_name
                    continue
                content = delta.tool_calls[0].function.arguments
                tool_calls_info[tool_id]["arguments"] += content

            if chunk.choices[0].delta.role:
                role = chunk.choices[0].delta.role
            
            if hasattr(chunk,"usage"):
                usage = Usage(completion_tokens=chunk.usage.completion_tokens,
                              prompt_tokens=chunk.usage.prompt_tokens,
                              total_tokens=chunk.usage.total_tokens)

            stream_response = StreamResponse(
                call_id=chunk.id,
                agent_name=self.agent_name,
                role=role,
                content=content,
                is_tool_call=bool(tool_id),
                tool_name=tool_name,
                is_tool_exec=False,
                tool_id=tool_id,
                usage=usage if hasattr(chunk,"usage") else None
            )
            self.stream_responses.append(stream_response)
            yield stream_response


        if current_content:
            self._add_message(role="assistant", content=current_content)

        tool_calls = [
            ToolCall(
                index=tool_info["index"] if "index" in tool_info else i,
                id=id,
                function=FunctionCall(
                    arguments=tool_info["arguments"], name=tool_info["name"]
                ),
                type="function",
            )
            for i, (id, tool_info) in enumerate(tool_calls_info.items())
        ]

        if tool_calls:
            complete_task = False
            for tool in tool_calls:
                if tool.function.name == "complete_task":
                    complete_task = True
                self.tools_mem.append(tool.model_dump())
            self._add_message(role="assistant", tool_calls=self.tools_mem)

            await self._execute_function_tools()
            self._enqueue_agent_tools()

            if complete_task:
                self.task_complete = True
                self.agents_queue.remove(self)

            async for stream_response in self.agents_queue[0].run(max_retries,n_call):
                yield stream_response


    async def _execute_function_tools(self):
        """Executes function tools in parallel."""
        tasks = []
        for tool in self.tools_mem:
            if tool["function"]["name"] in self.tools_functions:
                func = self.tools_functions[tool["function"]["name"]]
                tasks.append(self.execute_func_tool(tool, func))
        await asyncio.gather(*tasks)

    def _enqueue_agent_tools(self):
        """Enqueues agent tools for execution."""
        for tool in self.tools_mem:
            if tool["function"]["name"] in self.tools_agent:
                agent_method = self.tools_agent[tool["function"]["name"]]
                arguments = json.loads(tool["function"]["arguments"])
                agent_instance = agent_method(self, **arguments)
                agent_instance.agents_queue = self.agents_queue
                agent_instance.tool_call_id = tool["id"]
                agent_instance.tool_name = tool["function"]["name"]
                agent_instance.parent_agent = self
                # Ensure stream is False for child agents when parent is not streaming
                # if not self.completion_kwargs['stream']:
                #     agent_instance.stream = False
                # agent_instance.completion_kwargs['stream'] = self.completion_kwargs['stream']

                self.child_agents.append(agent_instance)
                self.agents_queue.appendleft(agent_instance)

    def add_message(self, role: str, content: Optional[str] = None, **kwargs) -> None:
        """
        Add a message to the conversation history of the first agent in the queue (current agent when called).
        """
        self.agents_queue[0].conv_history.messages.append(
            Message(
                role=role, agent_name=self.agents_queue[0].agent_name, content=content, **kwargs
            )
        )

    def _add_message(self, role: str, content: Optional[str] = None, **kwargs) -> None:
        """
        Add a message to the conversation history of the current agent.
        """
        self.conv_history.messages.append(
            Message(role=role, agent_name=self.agent_name, content=content, **kwargs)
        )

    async def execute_func_tool(self, tool: Dict[str, Any], func: Callable) -> None:
        """
        Asynchronously execute a single function tool and add the result to the conversation history.
        """
        try:
            arguments = json.loads(tool["function"]["arguments"])
            if asyncio.iscoroutinefunction(func):
                result = await func(self, **arguments)
            else:
                result = func(self, **arguments)
        except TypeError as e:
            result = f"Error executing tool {tool['function']['name']}: {str(e)}"
        except Exception as e:
            error_message = f"Unexpected error in tool execution: {str(e)}"
            self._handle_tool_error(error_message, tool, e)
            return

        self._add_message(
            role="tool",
            content=json.dumps(result),
            tool_call_id=tool["id"],
            tool_name=tool["function"]["name"],
        )

        if tool["function"]["name"] == "complete_task":
            self.parent_agent._add_message(
                role="tool",
                content=json.dumps(result),
                tool_call_id=self.tool_call_id,
                tool_name=self.tool_name,
            )
            self.parent_agent.tools_mem = [
                t for t in self.parent_agent.tools_mem if t["id"] != self.tool_call_id
            ]

        self.tools_mem = [t for t in self.tools_mem if t["id"] != tool["id"]]

    def _handle_tool_error(
        self, error_message: str, tool: Dict[str, Any], exception: Exception
    ):
        """
        Handle tool execution errors by logging and adding an error message to the conversation.
        """
        full_error = f"{error_message}\n\nStacktrace:\n{traceback.format_exc()}"
        self._add_message(
            role="tool",
            content=json.dumps({"error": error_message}),
            tool_call_id=tool["id"],
            tool_name=tool["function"]["name"],
        )
        # remove the tool from memory
        self.tools_mem = [t for t in self.tools_mem if t["id"] != tool["id"]]

class BaseTaskAgent(BaseAgent):
    tool_call_id: str = None
    tool_name: str = None

    def _discover_tools(self):
        """
        Override the tool discovery to add the complete_task tool schema.
        """
        tools, schemas = super()._discover_tools()
        if type(self).complete_task == BaseTaskAgent.complete_task:
            raise TypeError(
                f"\nERROR: {self.__class__.__name__} must implement the 'complete_task' method as tool.\n"
                "\nExample implementation:\n"
                "    @function_tool\n"
                "    def complete_task(self, result: str) -> Dict:\n"
                "        return {'status': 'success', 'result': result}\n"
            )

        return tools, schemas

    @function_tool
    def complete_task(
        self, result: str = Doc("The message to be returned to the user")
    ) -> Dict:
        """
        Abstract method that must be implemented by child classes.
        This method will be called when the task is complete.
        """
        pass