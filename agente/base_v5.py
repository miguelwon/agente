import json
from collections import defaultdict
import asyncio
from pydantic import BaseModel, Field
from typing import Deque,List, Optional, Dict, Union,Any,Callable, Tuple, AsyncGenerator,Annotated
from litellm import completion
from function_schema import get_function_schema,Doc
from functools import wraps
import traceback
from collections import deque


MAX_ITERATION = 6

class FunctionCall(BaseModel):
    arguments: str
    name: str

class ToolCall(BaseModel):
    index: int
    function: FunctionCall
    id: str
    type: str

# class Usage(BaseModel):
#     completion_tokens: int
#     prompt_tokens: int
#     total_tokens: int

class Message(BaseModel):
    role: str
    agent_name: str # Name of the agent that generated the message
    content: Optional[str] = Field(default=None)
    tool_calls: Optional[list[dict]] = Field(default=None) # List of tool calls. Each response can have multiple tool calls
    tool_call_id: Optional[str] = Field(default=None) # ID of the tool call in case the Message is for a tool call
    tool_name: Optional[str] = Field(default=None) # Name of the tool in case the Message is for a tool call
    hidden: Optional[bool] = Field(default=False) # Useful for hiding messages from the user
    id: Optional[str] = Field(default=None) # Unique ID for the message


class Response(BaseModel):
    call_id: str
    role: str
    content: str
    tool_calls: list[ToolCall] = Field(default=[])
    # usage: Usage = Field(default_factory=Usage)
    tool_executions: Optional[Dict[str, Any]] = Field(default=None)

# class StreamResponse(BaseModel):
#     call_id: str
#     role: str
#     content: str = None
#     is_tool_call: bool = False # whether the chunk if from a tool call
#     is_tool_exec: bool  = False # whether the chunk is from a tool execution
#     tool_id: str = None # ID of the tool call in case the chunk is from a tool call
#     usage: Optional[Usage] = Field(default=None)



class ConversationHistory(BaseModel):
    messages: list[Message]


class ToolExecutionError(Exception):
    """Custom exception for tool execution errors."""
    pass


def function_tool(func):
    """Decorator to mark a method as a function tool."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        except Exception as e:
            raise ToolExecutionError(f"Error in tool {func.__name__}: {str(e)}") from e
            
    wrapper.is_tool = True
    return wrapper

def agent_tool(cls):
    """Decorator that can only be used with classes that inherit from BaseTaskAgent."""
    if not isinstance(cls, type):
        raise TypeError("@agent_tool can only be used with classes. For methods, use @function_tool instead.")
    
    if not issubclass(cls, BaseTaskAgent):
        raise TypeError("@agent_tool can only be used with classes that inherit from BaseTaskAgent")
    
    cls.is_tool = True
    cls.is_agent = True
    cls.is_class_tool = True
    return cls


# def agent_tool(agent_cls):
#     """Decorator to mark a class as an agent tool."""
#     @wraps(agent_cls)
#     def wrapper(*args, **kwargs):
#         return agent_cls(*args, **kwargs)
    
#     wrapper.is_tool = True
#     wrapper.is_agent = True
#     return wrapper


def agent_tool(func):
    """Decorator to mark a method as an agent tool."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        except Exception as e:
            raise ToolExecutionError(f"Error in agent tool {func.__name__}: {str(e)}") from e
            
    wrapper.is_tool = True
    wrapper.is_agent = True  # Special flag to identify agent tools
    return wrapper

class BaseAgent:
    """
    A base class for AI agents with tool execution capabilities.
    """
    agent_name: str = None
    llm_model: str = None 
    stream: bool = None
    exec_tool: bool = None
    stream_tool_calls: bool = None
    stream_tool_exec: bool = None
    call_after_tool: bool = None


    def __init__(self, 
                agent_name: Optional[str] = None,
                llm_model:  Optional[str] = None,
                stream = False,
                exec_tool = True,
                stream_tool_calls = False,
                stream_tool_exec = False,
                call_after_tool = True,
                ):
        """
        Initialize the BaseAgent with automatic tool discovery.

        Args:
            agent_name: The name of the agent
            conv_history: The conversation history
        """
        self.agent_name = agent_name or self.__class__.agent_name
        self.llm_model = llm_model or self.__class__.llm_model
        self.stream = stream if stream is not None else self.__class__.stream
        self.exec_tool = exec_tool if exec_tool is not None else self.__class__.exec_tool
        self.stream_tool_calls = stream_tool_calls if stream_tool_calls is not None else self.__class__.stream_tool_calls
        self.stream_tool_exec = stream_tool_exec if stream_tool_exec is not None else self.__class__.stream_tool_exec
        self.call_after_tool = call_after_tool if call_after_tool is not None else self.__class__.call_after_tool

        if self.agent_name is None:
            raise ValueError("agent_name must be provided either as a class attribute or constructor parameter")
        if self.llm_model is None:
            raise ValueError("llm_model must be provided either as a class attribute or constructor parameter")


        if stream_tool_calls and not stream:
            raise ValueError("stream_tool_calls can only be True if stream is also True")
        if call_after_tool and not exec_tool:
            raise ValueError("call_after_tool can only be True if exec_tool is also True")
        if stream_tool_exec and not stream:
            raise ValueError("stream_tool_exec can only be True if stream is also True")



        self.conv_history = ConversationHistory(messages=[])
        self.all_responses: List[Response] = []
        self.tools, self.tools_schema = self._discover_tools()
        self.tool_functions = {tool.__name__: tool for tool in self.tools if not hasattr(tool, 'is_agent')}
        self.agent_functions = {tool.__name__: tool for tool in self.tools if hasattr(tool, 'is_agent')}

        self.tools_mem: List[Dict[str, Any]] = []
        self.logs_api_responses = [] 


        self.task_complete = False

        

    def _discover_tools(self) -> Tuple[List[Callable], List[Dict[str, Any]]]:
        """
        Discover and register tools (methods) marked with the @tool decorator.
        """
        tools = []
        schemas = []
        for name, method in vars(self.__class__).items():
            if callable(method) and getattr(method, 'is_tool', False):
                tools.append(getattr(self, name))
                schema = get_function_schema(method)


                if 'parameters' in schema and 'properties' in schema['parameters']:
                    schema['parameters']['properties'].pop('self', None)
                    if 'required' in schema['parameters']:
                        schema['parameters']['required'] = [param for param in schema['parameters']['required'] if param != 'self']
                schemas.append({"type": "function", "function": schema})
        
        return tools, schemas


    def add_message(self, role: str, content: Optional[str] = None, **kwargs) -> None:
        """
        Add a message to the conversation history.

        Args:
            role: The role of the message sender
            content: The message content
            **kwargs: Additional message attributes
        
        Returns:
            None
        """
        self.conv_history.messages.append(Message(role=role, agent_name=self.agent_name,content=content, **kwargs))
    
    async def run(self, 
                  model: Optional[str] = None,
                  stream: Optional[bool] = None,
                  exec_tool: Optional[bool] = None,
                  stream_tool_calls: Optional[bool] = None,
                  stream_tool_exec: Optional[bool] = None,
                  call_after_tool: Optional[bool] = None,
                  n_iteration: int = 1,
                  **completion_kwargs) -> Union[AsyncGenerator[str, None], str]:
        """
        Run the agent asynchronously, processing messages and executing tools.

        Args:
            model: The model to use for completion
            stream: Whether to stream the response
            exec_tool: Whether to execute tools
            stream_tool_calls: Whether to stream tool calls itself
            stream_tool_exec: Whether to stream tool execution (useful when a tool is an agent)
            call_after_tool: Whether to call the agent again after a tool call execution
            n_iteration: Number of iteration to keep track of inner loop
            **completion_kwargs: Additional completion parameters

        Returns:
            The response or an async generator of responses 
        """

        
        # Use instance attributes as defaults if parameters are None
        model = model if model is not None else self.llm_model
        stream = stream if stream is not None else self.stream
        exec_tool = exec_tool if exec_tool is not None else self.exec_tool
        stream_tool_calls = stream_tool_calls if stream_tool_calls is not None else self.stream_tool_calls
        stream_tool_exec = stream_tool_exec if stream_tool_exec is not None else self.stream_tool_exec
        call_after_tool = call_after_tool if call_after_tool is not None else self.call_after_tool



        if self.tools_mem:
            raise ValueError("There are still tools in memory to be executed")
        if stream_tool_calls and not stream:
            raise ValueError("stream_tool_calls can only be True if stream is also True")
        if call_after_tool and not exec_tool:
            raise ValueError("call_after_tool can only be True if exec_tool is also True")
        if stream_tool_exec and not stream:
            raise ValueError("stream_tool_exec can only be True if stream is also True")


        print("passou run",n_iteration,"|",self.agent_name)

        if n_iteration > MAX_ITERATION:
            if stream:
                async def iteration_limit_generator():
                    yield "Maximum number of iterations reached"
                return iteration_limit_generator()
            return "Maximum number of iterations reached"
    
        completion_params = {
            "model": model,
            "stream": stream,
            "messages": [message.dict(exclude_unset=True,exclude={'agent_name'}) for message in self.conv_history.messages if message.agent_name == self.agent_name],
            "tools": self.tools_schema,
            "tool_choice": "auto",
        }
        completion_params.update(completion_kwargs)

        # if completion_params['stream']:
        #     return self._run_stream(completion_params, exec_tool, stream_tool_calls,stream_tool_exec,call_after_tool,n_iteration)
        # else:
        return await self._run_no_stream(completion_params, exec_tool, call_after_tool,n_iteration)



    async def _run_no_stream(self, 
                             completion_params: dict, 
                             exec_tool: bool,
                             call_after_tool: bool,
                             n_iteration: int) -> Response:
        """
        Run the agent asynchronously without streaming the response.

        Args:
            completion_params: The completion parameters
            exec_tool: Whether to execute tools
            call_after_tool: Whether to call the agent again after a tool call execution
            n_iteration: Number of iteration to keep track of inner loop

        Returns:
            The response string
        """
        print("completion_params",completion_params)
        _response = completion(**completion_params)
        self.logs_api_responses.append(_response)
        content = _response.choices[0].message.content
        call_id = _response.id
        tool_calls = []
        if _response.choices[0].message.tool_calls:
            for tool_call in _response.choices[0].message.tool_calls:
                index = tool_call.index
                id = tool_call.id
                name = tool_call.function.name
                arguments = tool_call.function.arguments

                tool = ToolCall(index=index, id=id, name=name, function = FunctionCall(arguments=arguments, name=name),type = 'function')

                tool_calls.append(tool)


        # response = Response(call_id=call_id, role='assistant', content=content, tool_calls=tool_calls)


        if content:
            self.add_message(role='assistant', content=content)
        if tool_calls:
            for tool in tool_calls:
                self.tools_mem.append(tool.dict())
            self.add_message(role="assistant", tool_calls=self.tools_mem)
            if exec_tool:
                result = await self.execute_tools()

                if self.task_complete:
                    return result
        # check if is the final output of a task agent (if so, we must stop here)
        # if self.conv_history.messages[-1].role == "tool" and self.conv_history.messages[-1].tool_name == "final_output":
        #     return []


        print("ASASD",self.agent_name,self.conv_history.messages[-1])

        # if tools were called and executed, and call_after_tool is True, call the agent again
        if call_after_tool and self.conv_history.messages[-1].role == "tool":
            n_iteration = n_iteration + 1
            _completion_params = {k: v for k, v in completion_params.items() if k not in ['model', 'stream']}
            _completion_params['messages'] = [message.dict(exclude_unset=True,exclude={'agent_name'}) for message in self.conv_history.messages if message.agent_name == self.agent_name]
            return await self.run(
                                    model = completion_params['model'],
                                    stream= False,
                                    exec_tool=exec_tool, 
                                    stream_tool_calls=False, 
                                    stream_tool_exec=False,
                                    call_after_tool=True, 
                                    n_iteration=n_iteration,
                                    **_completion_params
                                  )
        return self.all_responses
    

    async def execute_tools(self) -> None:
        """
        Asynchronously execute all tools in the tools memory.
        """
        tasks = []
        for tool in self.tools_mem:
            if tool['function']['name'] in self.tool_functions:
                task = self.execute_func_tool(tool)
                tasks.append(task)

            elif tool['function']['name'] in self.agent_functions:
                print("passou gather task agent")
                task = self.execute_agent_tool(tool)
                tasks.append(task)


        await asyncio.gather(*tasks)
        self.tools_mem.clear()

        

    async def execute_func_tool(self, tool: Dict[str, Any]) -> None:
        """
        Asynchronously execute a single tool and add the result to the conversation history.
        Enhanced error handling provides more detailed error messages.

        Args:
            tool: The tool to execute

        Returns:
            None
        """
        try:
            arguments = json.loads(tool['function']['arguments'])
            function_name = tool['function']['name']
            func = self.tool_functions.get(function_name)
            if func is None:
                raise ValueError(f"Function {function_name} not found in available tools")
            print("**arguments",arguments)
            if asyncio.iscoroutinefunction(func):
                result = await func(**arguments)
            else:
                result = func(**arguments)

            print(">>>>>>",result,"tool_id",tool['id'])
            
            self.add_message(
                role="tool", 
                content=json.dumps(result), 
                tool_call_id=tool['id'], 
                tool_name=function_name)

        except json.JSONDecodeError as e:
            error_message = f"Invalid JSON in tool arguments: {str(e)}"
            self._handle_tool_error(error_message, tool, e)
        except ValueError as e:
            error_message = str(e)
            self._handle_tool_error(error_message, tool, e)
        except ToolExecutionError as e:
            error_message = str(e)
            self._handle_tool_error(error_message, tool, e)
        except Exception as e:
            error_message = f"Unexpected error in tool execution: {str(e)}"
            self._handle_tool_error(error_message, tool, e)


    async def execute_agent_tool(self, tool: Dict[str, Any]) -> None:
        """
        Asynchronously execute an agent tool and add the result to the conversation history.
        """
        try:
            arguments = json.loads(tool['function']['arguments'])
            agent_name = tool['function']['name']
            agent_tool = self.agent_functions.get(agent_name)
            
            if agent_tool is None:
                raise ValueError(f"Agent {agent_name} not found in available agents")

            # force the update of tool_call_id value 
            arguments['tool_call_id'] = tool['id']
            print("**arguments_ag",arguments)

            # Execute the agent and get its response
            result = await agent_tool(**arguments)

            print("result ",self.agent_name,result)
            
            # # Add the agent's response to the conversation history
            self.add_message(
                role="tool",
                content=json.dumps(result) if isinstance(result, (dict, list)) else str(result),
                tool_call_id=tool['id']
            )

        except json.JSONDecodeError as e:
            error_message = f"Invalid JSON in agent tool arguments: {str(e)}"
            self._handle_tool_error(error_message, tool, e)
        except Exception as e:
            error_message = f"Error executing agent tool {agent_name}: {str(e)}"
            self._handle_tool_error(error_message, tool, e)


    def _handle_tool_error(self, error_message: str, tool: Dict[str, Any], exception: Exception):
        """
        Handle tool execution errors by logging and adding an error message to the conversation.

        Args:
            error_message: The error message
            tool: The tool that caused the error
            exception: The exception that was raised    

        Returns:
            None        
        """
        full_error = f"{error_message}\n\nStacktrace:\n{traceback.format_exc()}"
        print(f"Error executing tool {tool['function']['name']}: {full_error}")
        self.add_message(
            role="tool", 
            content=json.dumps({"error": error_message}), 
            tool_call_id=tool['id'], 
            tool_name=tool['function']['name']
        )




class BaseTaskAgent(BaseAgent):

    def __init__(self, tool_call_id: str, **kwargs):
        """
        Initialize the BaseTaskAgent with required tool_call_id parameter.

        Args:
            tool_call_id: Required ID to track the tool call
            **kwargs: Additional parameters passed to BaseAgent
        """
        super().__init__(**kwargs)
        self.tool_call_id = tool_call_id


    def _discover_tools(self):
        """
        Override the tool discovery to ensure the complete_task tool exists.
        """
        tools, schemas = super()._discover_tools()
        
        has_complete_task = any(
            tool.__name__ == "complete_task" 
            for tool in tools
        )
        
        if not has_complete_task:
            raise ValueError(
                "TaskAgent requires a @function_tool decorated 'complete_task' method "
                "to be defined. This tool will be called when the task is complete."
            )
            
        return tools, schemas

    @function_tool
    def complete_task(self, 
                      result: Annotated[str, Doc("The message to be returned to the user with the joke")]
                      ) -> dict:
        """
        Default implementation of the task completion tool.
        Can be overridden by child classes for custom behavior.
        
        Args:
            result: The final result of the task
            
        Returns:
            The processed result
        """
        return {'tool_call_id': self.tool_call_id, 'result': result}



    # @function_tool
    # def complete_task(self, 
    #                  joke: Annotated[str, Doc("The message to be returned to the user with the joke")]
    #                  ) -> str:
    #     """Call this function when the task is complete with the final output as the argument."""
    #     return {'joke': joke}

    async def execute_tools(self) -> None:
        """
        Override execute_tools to check for task completion after tool execution.
        """
        # Store the current message count before executing tools
        initial_message_count = len(self.conv_history.messages)
        
        # Execute tools as normal
        await super().execute_tools()
        
        # Check all new messages added during tool execution
        new_messages = self.conv_history.messages[initial_message_count:]
        for msg in new_messages:
            if msg.role == "tool" and msg.tool_name == "complete_task":
                # If the complete_task tool was called, process the result
                result = json.loads(msg.content)

                print("!!!!",self.tool_call_id,result)
                self.task_complete = True

                return result['result']

        return True