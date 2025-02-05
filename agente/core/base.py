import asyncio
import json
import traceback
import warnings
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from litellm import acompletion
from pydantic import BaseModel, Field

from langchain_core.tools.base import create_schema_from_function
from langchain_core.utils.function_calling import convert_to_openai_function

from agente.models.schemas import (
    Message, Response, StreamResponse, ConversationHistory,
    ToolCall, FunctionCall,Usage
)
from .decorators import function_tool

from inspect import signature
import secrets

import ast

def gen_tool_id():
    """Generate a unique tool ID of 24 characters (no special characters)."""
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return ''.join(secrets.choice(chars) for _ in range(24))


def ensure_self_in_function_code(function_code: str, function_name: str) -> str:
    """
    Parse the function_code string, locate the function definition with name `function_name`,
    and ensure that its parameter list includes 'self'. If not, add 'self' as the first parameter.
    Return the modified code as a string.
    """
    tree = ast.parse(function_code)
    # Look for the function definition with the given name.
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # If no arguments or the first is not 'self', insert it.
            if not node.args.args or node.args.args[0].arg != 'self':
                self_arg = ast.arg(arg='self', annotation=None)
                node.args.args.insert(0, self_arg)
            break  # Assuming only one such function; otherwise, adjust as needed.

    modified_code = ast_unparse(tree)


    return modified_code



class AgentState:
    READY = "ready"
    WAITING_FOR_TOOLS = "waiting_for_tools"
    WAITING_FOR_USER = "waiting_for_user"
    COMPLETE = "complete"

class BaseAgent(BaseModel):
    """
    A base class for AI agents with tool execution capabilities.
    """

    agent_name: str
    system_prompt: Optional[str] = None
    is_conversational: bool = True
    parent_agent: Optional["BaseAgent"] = None
    child_agents: List["BaseAgent"] = Field(default_factory=list)
    conv_history: ConversationHistory = Field(
        default_factory=lambda: ConversationHistory(messages=[])
    )
    tools_mem: List[Dict[str, Any]] = Field(default_factory=list)
    tools_mem_temp: List[Dict[str, Any]] = Field(default_factory=list)
    log_calls: List[Any] = Field(default_factory=list)
    logs_completions: List[Any] = Field(default_factory=list)
    retry_count: int = Field(default=0)  # Add this field

    responses: List[Response] = Field(default_factory=list)
    stream_responses: List[StreamResponse] = Field(default_factory=list)
    agents_queue: Optional[Deque["BaseAgent"]] = Field(None, exclude=True)

    # These will be populated during model_post_init
    tools: List[Callable] = Field(default_factory=list, exclude=True)
    tools_schema: List[Dict[str, Any]] = Field(default_factory=list, exclude=True)
    tools_functions: Dict[str, Callable] =Field(default_factory=dict, exclude=True)
    tools_agent: Dict[str, Callable] = Field(default_factory=dict, exclude=True)


    completion_kwargs: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "arbitrary_types_allowed": True
    }
    

    state: str = Field(default=AgentState.READY)

    next_tool_map: Optional[Dict] = Field(default={}, description="Mapping of next tools to be called")
    manual_call_map: Optional[Dict] = Field(default={}, description="Mapping of manual calls from to next tools")
    orig_tool_choice: Optional[str] = Field(default=None, description="The original tool choice before the next_tool")

    can_add_tools: bool = False
    

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

        if self.system_prompt:
            self._add_message("system", content=self.system_prompt)

        if "tool_choice" in self.completion_kwargs:
            self.orig_tool_choice = self.completion_kwargs["tool_choice"]


    def _discover_tools(self) -> Tuple[List[Callable], List[Dict[str, Any]]]:
        """
        Discover and register tools (methods) marked with the @tool decorator.
        """
        tools = []
        schemas = []
        seen_names = set()

        for cls in reversed(self.__class__.__mro__):
            for name, method in vars(cls).items():
                if not self.can_add_tools and name == "add_tool":
                    continue

                if (
                    name == "complete_task"
                    and cls is BaseTaskAgent
                    and self.__class__ is not BaseTaskAgent
                ):
                    continue

                if name in seen_names:
                    continue

                if callable(method) and getattr(method, "is_tool", False):

                    ignored_params = getattr(method, "ignored_params", [])
                    if 'self' not in ignored_params:
                        ignored_params.append('self')
                    tools.append(method)
            
                    schema = create_schema_from_function(
                        method.__name__,
                        method,
                        filter_args=ignored_params,
                        parse_docstring=True,
                        error_on_invalid_docstring=False,
                        include_injected=True,
                    )
                    schema = convert_to_openai_function(schema.schema())

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


    # IMPORTANT: This feature is experimental and should be used with caution as it can lead to security vulnerabilities.
    @function_tool
    def add_tool(self, function_name:str,function_code:str, docstring:str) -> None:
        """Add a python function (tool) to the list of tools. Example:

        Args:
            function_name: The name of the function to be added to the list of tools.
            function_code: The python code of the function to be added to the list of tools. 
            docstring: The docstring of the tool function to be added. It must contain a description of the function and the parameters using "Agrs:" just like this docstring.
        """

        modified_function_code = ensure_self_in_function_code(function_code, function_name)


        namespace = {
            '__name__': self.__class__.__module__,  # Add module name to namespace
            '__file__': __file__
        }

        exec(modified_function_code, namespace)
        new_func = namespace[function_name]

        if not hasattr(new_func, 'is_tool'):
            new_func = function_tool(new_func)

        setattr(self.__class__, function_name, new_func)

        # Run tool discovery again to update the tools list.
        self.tools, self.tools_schema = self._discover_tools()

        print(f"Tool {function_name} added.")


        return "Tool added successfully"
 

    async def run(self, max_retries:int = 20) -> Any:
        """
        Run the agent asynchronously, processing messages and executing tools.
        """
        n_calls = 0
        
        while self.agents_queue[0].state == AgentState.READY and n_calls < max_retries:
            n_calls += 1
            current_agent = self.agents_queue[0] 
                
            # Handle tools in memory first
            if current_agent.tools_mem:
                current_agent._add_message(role="assistant", tool_calls=current_agent.tools_mem)
                await current_agent._execute_function_tools()
                current_agent._enqueue_agent_tools()
                n_calls -= 1
                continue
                
            # Prepare messages for completion
            messages = [
                message.model_dump(exclude_unset=True, exclude={"agent_name", "hidden", "id"})
                for message in current_agent.conv_history.messages
                if message.agent_name == current_agent.agent_name
            ]

            # Hack: handle message caching (only first two messages)
            # messages = [
            #     {**m, "content": [{"type":"text", "text":m["content"], "cache_control": {"type": "ephemeral"}}]} 
            #     if i < 2 else m
            #     for i, m in enumerate(messages)
            # ]

            # Check for next tool            
            # #check if tool was called because of tool_choice
            if self.next_tool_map:
                next_tool = list(self.next_tool_map.values())[0]
                self.completion_kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": next_tool}
                }
            else:
                #restore the original tool choice
                if self.orig_tool_choice is not None:
                    self.completion_kwargs["tool_choice"] = self.orig_tool_choice
                else:
                    if "tool_choice" in self.completion_kwargs:
                        del self.completion_kwargs["tool_choice"]


            # Prepare completion parameters
            completion_params = {
                **current_agent.completion_config,
                "messages": messages,
            }
            if current_agent.tools_schema:
                completion_params["tools"] = current_agent.tools_schema

            # Run completion with or without streaming
            print("Executing agent:",current_agent.agent_name)
            if completion_params["stream"]:
                async for response in current_agent._run_stream(completion_params):
                    yield response
            else:
                async for response in current_agent._run_no_stream(completion_params):
                    yield response


        if n_calls >= max_retries:
            warnings.warn(f"Max retries ({max_retries}) reached for agent {self.agent_name}")
            self.add_message("assistant", "Max retries reached")


    async def _run_no_stream(
        self,
        completion_params: Dict
    ):
        """
        Run the agent asynchronously without streaming the response.
        """
        task_complete = False
        self.log_calls.append(completion_params)
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
        else:
            usage = None


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
            for tool in tool_calls:
                if tool.function.name == "complete_task":
                    task_complete = True
                self.tools_mem.append(tool.model_dump())
            self._add_message(role="assistant", tool_calls=self.tools_mem)

            #first we execute the function tools    
            await self._execute_function_tools()

            #then we enqueue the agent tools
            self._enqueue_agent_tools()

            if task_complete:
                self.state = AgentState.COMPLETE
                self.agents_queue.remove(self)

            # add tool in the temporary memory to the main memory (must be done only aftert tool execution and agent enqueuing)
            if self.tools_mem_temp:                
                self.tools_mem.extend(self.tools_mem_temp)
                self.tools_mem_temp = []

        if not task_complete and not tool_calls:
            self.state = AgentState.WAITING_FOR_USER


    async def _run_stream(
        self,
        completion_params: Dict
    ):
        """
        Run the agent asynchronously with streaming the response.
        """
        tool_id = None
        tool_name = None
        role = "assistant"
        tool_calls_info = defaultdict(lambda: defaultdict(str))
        current_content = ""
        usage = None
        task_complete = False

        self.log_calls.append(completion_params)

        async for chunk in await acompletion(**completion_params):
            self.logs_completions.append(chunk)
            if hasattr(chunk,"usage"):
                usage = Usage(completion_tokens=chunk.usage.completion_tokens,
                              prompt_tokens=chunk.usage.prompt_tokens,
                              total_tokens=chunk.usage.total_tokens)

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
            if not usage:
                self._add_message(role="assistant", content=current_content)
            else:
                self._add_message(role="assistant", content=current_content, usage=usage)
            # usage = None # because at this point usage already accounts for the tokens used for the tool calls


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
            task_complete = False
            for tool in tool_calls:
                if tool.function.name == "complete_task":
                    task_complete = True
                self.tools_mem.append(tool.model_dump())

            if not usage:
                self._add_message(role="assistant", tool_calls=self.tools_mem)
            else:
                self._add_message(role="assistant", tool_calls=self.tools_mem, usage=usage)

            await self._execute_function_tools()
            self._enqueue_agent_tools()

            if task_complete:
                self.state = AgentState.COMPLETE
                self.agents_queue.remove(self)

            # add tool in the temporary memory to the main memory (must be done only aftert tool execution and agent enqueuing)
            if self.tools_mem_temp:                
                self.tools_mem.extend(self.tools_mem_temp)
                self.tools_mem_temp = []


        if not task_complete and not tool_calls:
            self.state = AgentState.WAITING_FOR_USER


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
                # print(f"{self.agent_name} called agent tool:",tool["function"]["name"])
                agent_method = self.tools_agent[tool["function"]["name"]]
                try:
                    arguments = json.loads(tool["function"]["arguments"])
                except json.JSONDecodeError:
                    print(f"Error decoding arguments for tool {tool['function']['name']}\n\n{tool['function']['arguments']}")
                    # TODO: handle this error

                agent_instance = agent_method(self, **arguments)
                agent_instance.agents_queue = self.agents_queue # inherit the queue
                agent_instance.tool_call_id = tool["id"] # set the tool call id so that when the task is complete, the result is sent to the parent agent with the correct tool id
                agent_instance.tool_name = tool["function"]["name"] # set the tool name so that when the task is complete, the result is sent to the parent agent with the correct tool name
                agent_instance.parent_agent = self # set the parent agent

   
                self.child_agents.append(agent_instance)
                self.agents_queue.appendleft(agent_instance)



    async def execute_func_tool(self, tool: Dict[str, Any], func: Callable) -> None:
        """
        Asynchronously execute a single function tool and add the result to the conversation history.
        """
        this_tool_name = tool["function"]["name"]
        print(f"Executing tool: {this_tool_name} from agent {self.agent_name}")

        try:
            arguments = json.loads(tool["function"]["arguments"])
            if asyncio.iscoroutinefunction(func):
                result = await func(self, **arguments)
            else:
                result = func(self, **arguments)

        except TypeError as e:
            result = f"Error executing tool {this_tool_name}: {str(e)}"
        except Exception as e:
            error_message = f"Unexpected error in tool execution: {str(e)}"
            self._handle_tool_error(error_message, tool, e)
            return

        self._add_message(
            role="tool",
            content=json.dumps(result),
            tool_call_id=tool["id"],
            tool_name=this_tool_name,
        )


        if this_tool_name == "complete_task":
            #If there means that the task is complete and the flow switches back to the parent agent

            self.parent_agent._add_message(
                role="tool",
                content=json.dumps(result),
                tool_call_id=self.tool_call_id,
                tool_name=self.tool_name,
            )

            agent_method = self.parent_agent.tools_agent[self.tool_name]
            next_tool = getattr(agent_method, "next_tool", None)
            manual_call = getattr(agent_method, "manual_call", None)
            if next_tool:
                self._add_next_tool(self.parent_agent, self.tool_name,next_tool, manual_call,result)
            if self.tool_name in self.parent_agent.next_tool_map:
                self._add_manual_tool_call(self.parent_agent,self.tool_name,result)

            self._cleanup_tool_map(self.parent_agent, self.tool_name)

            #clean the tool from the parent agent
            self.parent_agent.tools_mem = [
                t for t in self.parent_agent.tools_mem if t["id"] != self.tool_call_id
            ]

        else:

            #If here means that the tool was executed and the flow continues within the same agent (where is TaskAgent or BaseAgent)
            tool_method = self.tools_functions[this_tool_name]
            next_tool = getattr(tool_method, "next_tool", None)
            manual_call = getattr(func, "manual_call", None)
            if next_tool:
                self._add_next_tool(self, this_tool_name,next_tool,manual_call,result)
            if this_tool_name in self.next_tool_map:
                self._add_manual_tool_call(self,this_tool_name,result)

            self._cleanup_tool_map(self, this_tool_name)


        self.tools_mem = [t for t in self.tools_mem if t["id"] != tool["id"]]

    def _add_next_tool(self, agent, current_tool,next_tool, manual_call,result):
        """Add the next tool to the next_tool_map and manual_call_map."""    
        agent.next_tool_map[current_tool] = next_tool
        if manual_call:
            agent.manual_call_map[next_tool] = manual_call
    
    def _add_manual_tool_call(self,agent,current_tool,result):        
        next_tool = agent.next_tool_map[current_tool]
        if next_tool in agent.manual_call_map:
            manual_call = agent.manual_call_map[next_tool]

            manual_args = manual_call(result)
            tools_dict = agent.tools_agent if next_tool in agent.tools_agent else agent.tools_functions
            sig_params = set(signature(tools_dict[next_tool]).parameters.keys()) - {'self'}
            
            if sig_params == set(manual_args.keys()):
                manual_args = json.dumps(manual_args)
                manual_tool_call = {
                    "id": f"man_{gen_tool_id()}",
                    "function": {
                        "name": next_tool,
                        "arguments": manual_args
                    },
                    "type": "function"
                }
                agent.tools_mem_temp.append(manual_tool_call)
            else:
                warnings.warn(f"Manual call: arguments for {next_tool} are not compatible with the function signature")

    def _cleanup_tool_map(self, agent, tool_name):
        """Helper method to clean up completed tools from the next_tool_map."""
        if agent.next_tool_map:
            tool_names = [v for v in agent.next_tool_map.values()]
            if tool_name in tool_names:
                agent.next_tool_map = {k:v for k,v in agent.next_tool_map.items() if v != tool_name}


    def add_message(self, role: str, content: Optional[str] = None, **kwargs) -> None:
        """
        Add a message to the conversation history of the first agent in the queue (current agent when called).
        """
        self.agents_queue[0].conv_history.messages.append(
            Message(
                role=role, agent_name=self.agents_queue[0].agent_name, content=content, **kwargs
            )
        )
        self.agents_queue[0].state = AgentState.READY

    def _add_message(self, role: str, content: Optional[str] = None, **kwargs) -> None:
        """
        Add a message to the conversation history of the self agent.
        """
        self.conv_history.messages.append(
            Message(role=role, agent_name=self.agent_name, content=content, **kwargs)
        )

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
    def complete_task(self, result:str) -> Dict:
        """
        Abstract method that must be implemented by child classes.
        This method will be called when the task is complete.

        Args:
            result: The message to be returned to the parent agent that called the task.
        """
        pass
