"""Base agent implementation for the Agente framework."""
import asyncio
import json
import traceback
import warnings
from collections import defaultdict, deque
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Type, Union, Literal, Annotated
import logging

# Third-party imports
from litellm import acompletion
from pydantic import BaseModel, Field, validator
from langchain_core.tools.base import create_schema_from_function
from langchain_core.utils.function_calling import convert_to_openai_function

# Local imports
from agente.models.schemas import (
    Message, Response, StreamResponse, ConversationHistory,
    ToolCall, FunctionCall, Usage, Content, ThinkingBlock,
    ContentThinking, ContentRedactedThinking, ContentUnion
)
from .decorators import function_tool
from .exceptions import (
    AgentExecutionError, ToolExecutionError, MaxRetriesExceededError,
    InvalidToolError,StreamingMismatchError
)


# Configure logging
logger = logging.getLogger(__name__)


def ensure_self_in_function_code(function_code: str, function_name: str) -> str:
    """
    Ensures that the function code has 'self' as the first parameter.
    Uses AST parsing for robustness.
    
    Args:
        function_code: The original function code
        function_name: The name of the function
        
    Returns:
        Modified function code with 'self' as first parameter
    """
    import ast
    import sys
    
    try:
        # Parse the function code
        tree = ast.parse(function_code)
        
        # Look for the function definition with the given name
        modified = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Check if 'self' is already the first parameter
                if not node.args.args or node.args.args[0].arg != 'self':
                    # Create 'self' argument
                    if sys.version_info >= (3, 9):
                        self_arg = ast.arg(arg='self', annotation=None)
                    else:
                        self_arg = ast.arg(arg='self', annotation=None)
                    
                    # Insert 'self' as first parameter
                    node.args.args.insert(0, self_arg)
                    modified = True
                break
        
        if not modified:
            return function_code
        
        # Convert back to string
        if sys.version_info >= (3, 9):
            return ast.unparse(tree)
        else:
            # For older Python versions, fall back to a simple approach
            # or use astor if available
            try:
                import astor
                return astor.to_source(tree)
            except ImportError:
                # Fallback to regex approach for older Python versions
                import re
                func_pattern = rf'def\s+{function_name}\s*\((.*?)\)'
                match = re.search(func_pattern, function_code, re.DOTALL)
                
                if match:
                    params = match.group(1).strip()
                    if not params or not params.startswith('self'):
                        if params:
                            new_params = f"self, {params}"
                        else:
                            new_params = "self"
                        new_def = f"def {function_name}({new_params})"
                        function_code = re.sub(func_pattern, new_def, function_code, count=1)
                
                return function_code
                
    except SyntaxError:
        # If AST parsing fails, return original code
        logger.warning(f"Failed to parse function code for {function_name}, returning original")
        return function_code


class AgentState:
    READY = "ready" # Agent is ready to execute a task
    WAITING_FOR_TOOLS = "waiting_for_tools" # Agent is waiting for tools to be called
    WAITING_FOR_USER = "waiting_for_user" # Agent is waiting for user input
    COMPLETE = "complete" # Agent has completed its task (for task agents)



class ToolRegistry:
    """Manages tool discovery and registration."""
    
    def __init__(self, agent: 'BaseAgent'):
        self.agent = agent
        self.tools: List[Callable] = []
        self.tools_schema: List[Dict[str, Any]] = []
        self.tools_functions: Dict[str, Callable] = {}
        self.tools_agent: Dict[str, Callable] = {}
    
    def discover_tools(self) -> Tuple[List[Callable], List[Dict[str, Any]]]:
        """Discover and register tools from the agent class."""
        tools = []
        schemas = []
        seen_names = set()
        
        for cls in reversed(self.agent.__class__.__mro__):
            for name, method in vars(cls).items():
                if self._should_skip_method(name, cls, seen_names):
                    continue
                
                if callable(method) and getattr(method, "is_tool", False):
                    tools.append(method)
                    schema = self._create_tool_schema(method)
                    schemas.append({"type": "function", "function": schema})
                    seen_names.add(name)

        self.tools = tools
        self.tools_schema = schemas
                
        self._categorize_tools(tools)
        return tools, schemas
    
    def _should_skip_method(self, name: str, cls: type, seen_names: set) -> bool:
        """Check if a method should be skipped during discovery."""
        if name in seen_names:
            return True
        
        if not self.agent.can_add_tools and name == "add_tool":
            return True
        
        if (name == "task_completed" and 
            cls.__name__ == "BaseTaskAgent" and 
            self.agent.__class__.__name__ != "BaseTaskAgent"):
            return True
        
        return False
    
    def _create_tool_schema(self, method: Callable) -> Dict[str, Any]:
        """Create OpenAI-compatible schema for a tool method."""
        ignored_params = getattr(method, "ignored_params", [])
        if 'self' not in ignored_params:
            ignored_params.append('self')
        
        schema = create_schema_from_function(
            method.__name__,
            method,
            filter_args=ignored_params,
            parse_docstring=True,
            error_on_invalid_docstring=False,
            include_injected=True,
        )
        return convert_to_openai_function(schema.schema())
    
    def _categorize_tools(self, tools: List[Callable]) -> None:
        """Categorize tools into function tools and agent tools."""
        self.tools_functions = {
            tool.__name__: tool
            for tool in tools
            if not getattr(tool, "is_agent", False)
        }
        self.tools_agent = {
            tool.__name__: tool
            for tool in tools
            if getattr(tool, "is_agent", False)
        }


class BaseAgent(BaseModel):
    """
    Base class for AI agents with tool execution capabilities.
    
    This class provides the foundation for creating AI agents that can:
    - Execute tools (methods decorated with @function_tool)
    - Manage conversation history
    - Handle streaming and non-streaming responses
    - Coordinate with child agents
    """
    
    # Agent configuration
    agent_name: str = Field(..., description="Unique name for the agent")
    system_prompt: Optional[str] = Field(None, description="System prompt for the agent")
    is_conversational: bool = Field(True, description="Whether agent is conversational or simple call and return interaction")
    
    # Agent hierarchy
    parent_agent: Optional["BaseAgent"] = Field(None, description="Parent agent if this is a child")
    child_agents: List["BaseAgent"] = Field(default_factory=list, description="List of child agents")
    
    # Conversation and memory
    conv_history: ConversationHistory = Field(
        default_factory=lambda: ConversationHistory(messages=[]), 
        description="Conversation history of the agent"
    )
    tools_mem: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Tools to be called (not in temporary memory)"
    )
    # Logging
    log_calls: List[Any] = Field(default_factory=list, description="Log of API calls made")
    logs_completions: List[Any] = Field(default_factory=list, description="Log of API responses")
    
    # Response tracking
    responses: List[Response] = Field(default_factory=list, description="Non-streaming responses")
    stream_responses: List[StreamResponse] = Field(default_factory=list, description="Streaming responses")
    
    # Agent queue and state
    agents_queue: Optional[Deque["BaseAgent"]] = Field(
        None, exclude=True, description="Main agent execution queue"
    )
    agents_queue_history: List[Deque["BaseAgent"]] = Field(
        default_factory=list, description="History of agent queue states"
    )
    state: AgentState = Field(default=AgentState.READY, description="Current agent state")
    
    # Tool management
    tool_registry: Optional[ToolRegistry] = Field(
        None, exclude=True, description="Registry for managing agent tools"
    )

    next_tool_forced: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list, description="List of tools to be called in the next call (forced execution)"
    )

    orig_tool_choice: Optional[Union[str, Dict]] = Field(
        None, 
        description="Original tool choice to restore after forced tool execution"
    )
    
    # Configuration
    completion_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters for LLM completion calls"
    )
    can_add_tools: bool = Field(False, description="Whether dynamic tool addition is allowed")
    max_calls_safeguard: int = Field(30, description="Maximum number of calls to prevent infinite loops")
    retry_count: int = Field(0, description="Retry count")
    max_retries: int = Field(20, description="Maximum retry attempts")
    silent: bool = Field(False, description="Whether to suppress console output")
    error_verbosity: str = Field(
        "normal", 
        description="""Error verbosity level:
        - 'minimal': Only show basic error notifications ('Error in tool_name')
        - 'normal': Show clean error messages with exception details (default)
        - 'verbose': Show full error messages with tracebacks"""
    )
    
    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
    
    @validator('agent_name')
    def validate_agent_name(cls, v):
        """Ensure agent name is not empty and contains valid characters."""
        if not v or not v.strip():
            raise ValueError("Agent name cannot be empty")
        if not v.replace('_', '').replace('-', '').replace(' ', '').isalnum():
            raise ValueError("Agent name can only contain alphanumeric characters, spaces, hyphens and underscores")
        return v
    
    def model_post_init(self, __context) -> None:
        """Initialize agent after Pydantic validation."""
        try:
            # Set up completion configuration
            self._setup_completion_config()
            
            # Initialize tool registry and discover tools
            self.tool_registry = ToolRegistry(self)
            self.tool_registry.discover_tools()
            # Initialize agent queue
            self._initialize_queue()
            
            # Add system prompt if provided
            if self.system_prompt:
                self._add_message("system", content=self.system_prompt)
            
            # Store original tool choice
            if "tool_choice" in self.completion_kwargs:
                self.orig_tool_choice = self.completion_kwargs["tool_choice"]
                
        except Exception as e:
            logger.error(f"Failed to initialize agent {self.agent_name}: {e}")
            raise AgentExecutionError(f"Agent initialization failed: {e}")
    
    def _setup_completion_config(self) -> None:
        """Set up completion configuration with defaults."""
        default_kwargs = {
            "model": "gpt-4.1",
            "stream": False,
            "max_tokens": None,
        }
        self.completion_kwargs = {**default_kwargs, **self.completion_kwargs}
    
    def _initialize_queue(self) -> None:
        """Initialize the agent queue."""
        if self.agents_queue is None:
            self.agents_queue = deque()
        
        if not self.parent_agent:
            self.agents_queue.appendleft(self)
            self.agents_queue_history.append(self.agents_queue[0])
    
    @property
    def tools(self) -> List[Callable]:
        """Get registered tools."""
        return self.tool_registry.tools if self.tool_registry else []
    
    @property
    def tools_schema(self) -> List[Dict[str, Any]]:
        """Get tool schemas."""
        return self.tool_registry.tools_schema if self.tool_registry else []
    
    @property
    def tools_functions(self) -> Dict[str, Callable]:
        """Get function tools mapping."""
        return self.tool_registry.tools_functions if self.tool_registry else {}
    
    @property
    def tools_agent(self) -> Dict[str, Callable]:
        """Get agent tools mapping."""
        return self.tool_registry.tools_agent if self.tool_registry else {}
    
    # IMPORTANT: This feature is experimental and should be used with caution as it can lead to security vulnerabilities.
    @function_tool
    def add_tool(self, 
                    function_name: Annotated[str, "The name of the function to be added to the list of tools."],
                    function_code: Annotated[str, "The python code of the function to be added to the list of tools, with a docstring"]) -> str:
        """ Call this tool to add a new tool (a python function) to the main list of available tools. Example:

function_name = "calculate_circle_area"
function_code = '''
def calculate_circle_area(self, radius: float) -> float:
    \"\"\"Calculate the area of a circle given its radius.

    Args:
        radius: The radius of the circle (must be a positive number).
    \"\"\"
    import math
    return math.pi * radius ** 2
'''
"""

        if not self.can_add_tools:
            return "Error: This agent is not configured to add tools dynamically. Set can_add_tools=True to enable this feature."
        
        
        # Ensure self parameter is in the function code
        modified_function_code = ensure_self_in_function_code(function_code, function_name)

        # Create namespace for exec
        namespace = {
            '__name__': self.__class__.__module__,  # Add module name to namespace
            '__file__': __file__,
        }
        
        try:
            # Execute the function code
            exec(modified_function_code, namespace)
            new_func = namespace[function_name]
            
            # Apply function_tool decorator if not already applied
            if not hasattr(new_func, 'is_tool'):
                new_func = function_tool(new_func)
            
            # Add the function to the class
            setattr(self.__class__, function_name, new_func)
            
            # Rediscover tools to update the registry
            self.tool_registry.discover_tools()
            
            if not self.silent:
                print(f"Tool {function_name} added successfully.")
            
            return f"Tool {function_name} added successfully"
            
        except Exception as e:
            error_msg = f"Failed to add tool {function_name}: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"


    async def run(
        self, 
        max_retries: Optional[int] = None, 
        stream: Optional[bool] = None,
        output_format: Literal['agente', 'litellm'] = 'litellm'
    ) -> Union[List[Union[Response, StreamResponse]], Any]:
        """
        Run the agent asynchronously.
        
        Args:
            max_retries: Maximum number of retry attempts (overrides instance setting)
            stream: Whether to stream responses
                If True, returns an async generator that yields responses
                If False, returns a list of all responses
                If None, uses the current completion_kwargs["stream"] setting
            output_format: The format of the response ('agente' or 'litellm'). Defaults to 'litellm'.
            
        Returns:
            If stream=False: List of Response objects (if 'agente') or litellm ModelResponse objects (if 'litellm')
            If stream=True: Async generator yielding StreamResponse objects (if 'agente') or litellm stream chunks (if 'litellm')
            
        Raises:
            MaxRetriesExceededError: If maximum retries are exceeded
            AgentExecutionError: If agent execution fails
        """
        # Set streaming behavior if provided
        if stream is not None:
            self.completion_kwargs["stream"] = stream
        
        # Get the current streaming setting
        is_streaming = self.completion_kwargs.get("stream", False)
        
        if is_streaming:
            # For streaming, return the generator directly
            return self._run_generator(max_retries, output_format=output_format)
        else:
            # For non-streaming, collect all responses and return as list
            responses = []
            async for response in self._run_generator(max_retries, output_format=output_format):
                responses.append(response)
            return responses


    async def _run_generator(self, max_retries: Optional[int] = None, output_format: Literal['agente', 'litellm'] = 'litellm') -> Any:
        """
        Internal generator method that contains the original run logic.
        
        Args:
            max_retries: Maximum number of retry attempts (overrides instance setting)
            
        Yields:
            Response or StreamResponse objects
            
        Raises:
            MaxRetriesExceededError: If maximum retries are exceeded
            AgentExecutionError: If agent execution fails
        """
        max_retries = max_retries or self.max_retries
        n_calls = 0
        
        try:
            while (self.agents_queue and 
                self.agents_queue[0].state == AgentState.READY and 
                n_calls < max_retries):
                
                n_calls += 1
                current_agent = self.agents_queue[0]

                # First check if there are tools in the memory. 
                if current_agent.tools_mem:                    
                    await current_agent._process_memory_tools()
                    n_calls -= 1
                    continue

                # Prepare messages
                messages = self._prepare_messages(current_agent)
                
                # Now check if there are tools in the next_tool_forced. 
                if len(current_agent.next_tool_forced) > 0:
                    # update tool choice to the next tool in the forced list
                    forced_tool_name = current_agent.next_tool_forced[0]["tool_name"]
                    manual_call_args = current_agent.next_tool_forced[0]["manual_call_args"]

                    current_agent._update_tool_choice(forced_tool_name)
                    # pop the forced tool from the forced list
                    current_agent.next_tool_forced.pop(0)
                    if manual_call_args:
                        #create a manual tool call
                        manual_tool_call = current_agent._create_manual_tool_call(
                            current_agent,
                            forced_tool_name,
                            manual_call_args
                        )
                        if manual_tool_call:
                            current_agent.tools_mem.append(manual_tool_call)                        
                        continue
                else:
                    # update tool choice to the original tool choice (if there is one)
                    current_agent._update_tool_choice(self.orig_tool_choice)


                # Prepare completion parameters
                completion_params = current_agent._prepare_completion_params(messages)
                
                # Log execution
                if not current_agent.silent:
                    print(f"Executing agent: {current_agent.agent_name}")
                    logger.info(f"Executing agent: {current_agent.agent_name}")

                
                # Execute completion
                if completion_params["stream"]:
                    async for response in current_agent._run_stream(completion_params, output_format=output_format):
                        yield response
                else:
                    async for response in current_agent._run_no_stream(completion_params, output_format=output_format):
                        yield response

                
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            raise
        
        if n_calls >= max_retries:
            error_msg = f"Maximum retries ({max_retries}) exceeded for agent {self.agent_name}"
            logger.warning(error_msg)
            self.add_message("assistant", error_msg)
            raise MaxRetriesExceededError(error_msg)

    
    async def _process_memory_tools(self) -> None:
        """Process tools stored in memory."""
        self._add_message(role="assistant", tool_calls=self.tools_mem)
        await self._execute_function_tools()
        self._enqueue_agent_tools()
    
    
    def _prepare_messages(self,agent: 'BaseAgent') -> List[Dict[str, Any]]:
        """Prepare messages for completion."""
        messages = [
            message.model_dump(
                exclude_unset=True,
                exclude={"agent_name", "hidden", "id",'usage'}
            )
            for message in agent.conv_history.messages
            if message.agent_name == agent.agent_name
        ]
        return messages

    def _update_tool_choice(self,tool_name:str = None) -> None:
        """Update tool choice based on next_tool_forced. Runs always before calling the LLM"""
        if tool_name:
            self.completion_kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_name}
                }
        else:
            self.completion_kwargs.pop("tool_choice", None)

    
    def _prepare_completion_params(self, messages: List[Dict]) -> Dict[str, Any]:
        """Prepare parameters for completion API."""
        params = {
            **self.completion_kwargs,
            "messages": messages,
        }
        
        if self.tools_schema:
            params["tools"] = self.tools_schema
            
            # Validate tools
            for tool in params["tools"]:
                tool_name = tool["function"]["name"]
                if tool_name not in self.tools_functions and tool_name not in self.tools_agent:
                    raise InvalidToolError(f"Tool '{tool_name}' not found in registry")
        
        return params


    async def _run_no_stream(self, completion_params: Dict[str, Any], output_format: Literal['agente', 'litellm'] = 'agente') -> Any:
        """
        Execute completion without streaming.
        
        Args:
            completion_params: Parameters for the completion API
            output_format: The format of the response ('agente' or 'litellm').
            
        Yields:
            Response objects or litellm ModelResponse.
        """
        try:
            # Log the API call
            self.log_calls.append(completion_params)
            
            # Make the API call
            api_response = await acompletion(**completion_params)
            self.logs_completions.append(api_response)
            
            # Process the response
            response = self._process_api_response(api_response)
            self.responses.append(response)
            
            if output_format == 'litellm':
                yield api_response
            else:
                yield response
            
            # Handle message and tool calls
            await self._handle_response_content(response)
            
            # Update state based on the response
            self._update_agent_state_no_stream(response)
            

        except Exception as e:
            logger.error(f"Non-streaming completion failed: {e}")
            raise AgentExecutionError(f"Completion failed: {e}") from e


    async def _run_stream(self, completion_params: Dict[str, Any], output_format: Literal['agente', 'litellm'] = 'agente') -> Any:
        """
        Execute completion with streaming.
        
        Args:
            completion_params: Parameters for the completion API
            output_format: The format of the response ('agente' or 'litellm').
            
        Yields:
            StreamResponse objects or litellm stream chunks.
        """
        try:
            # Initialize streaming state
            stream_state = StreamingState()
            
            # Log the API call
            self.log_calls.append(completion_params)
            
            # Process stream chunks
            async for chunk in await acompletion(**completion_params):
                self.logs_completions.append(chunk)
                
                # Process chunk and yield response
                stream_response = self._process_stream_chunk(chunk, stream_state)
                if stream_response:
                    self.stream_responses.append(stream_response)

                if output_format == 'litellm':
                    yield chunk
                elif stream_response:
                    yield stream_response
            
            # Handle accumulated content after streaming
            await self._handle_stream_completion(stream_state)

            # Update state based on what happened during streaming
            self._update_agent_state_from_stream(stream_state)

                
        except Exception as e:
            logger.error(f"Streaming completion failed: {e}")
            raise AgentExecutionError(f"Streaming failed: {e}") from e
    
    def _update_agent_state_no_stream(self, response: Response) -> None:
        """
        Update agent state after processing a response. 
        Order of checks is important here.
        
        Args:
            response: The processed response
        """
        # If we have tool calls, we remain READY (tools will be executed)
        if response.tool_calls:
            self.state = AgentState.READY
            return

        # For conversational agents with no tools to execute
        if self.is_conversational:
            self.state = AgentState.WAITING_FOR_USER
            return

        # For task agents that haven't completed their task
        if isinstance(self, BaseTaskAgent) and not self.completed_task:
            self.state = AgentState.WAITING_FOR_USER
            return

        # For task agents that haven't completed their task
        if isinstance(self, BaseTaskAgent) and not self.completed_task:
            self.state = AgentState.WAITING_FOR_USER
            return        

        # Default: wait for user
        self.state = AgentState.WAITING_FOR_USER

    def _update_agent_state_from_stream(self, stream_state: 'StreamingState') -> None:
        """
        Update agent state after streaming completion.
        
        Args:
            stream_state: The final streaming state
        """
        # If we have tool calls, we remain READY
        if stream_state.tool_calls_info:
            self.state = AgentState.READY
            return

        # For conversational agents
        if self.is_conversational:
            self.state = AgentState.WAITING_FOR_USER
            return

        # For task agents that haven't completed their task
        if isinstance(self, BaseTaskAgent) and not self.completed_task:
            self.state = AgentState.WAITING_FOR_USER
            return

        # For task agents that have completed their task
        if isinstance(self, BaseTaskAgent) and self.completed_task:
            self.state = AgentState.COMPLETE
            return
                
        # Default: wait for user
        self.state = AgentState.WAITING_FOR_USER

    def _process_api_response(self, api_response: Any) -> Response:
        """
        Process API response into a Response object.
        
        Args:
            api_response: Raw API response
            
        Returns:
            Processed Response object
        """
        choice = api_response.choices[0]
        message = choice.message
        
        # Extract content
        content = message.content or ""
        
        # Extract tool calls
        tool_calls = []
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls = [
                ToolCall(
                    index=tc.index if hasattr(tc, 'index') else i,
                    id=tc.id,
                    function=FunctionCall(
                        arguments=tc.function.arguments,
                        name=tc.function.name
                    ),
                    type="function"
                )
                for i, tc in enumerate(message.tool_calls)
            ]
        
        # Extract usage
        usage = None
        if hasattr(api_response, 'usage'):
            usage = Usage(
                completion_tokens=api_response.usage.completion_tokens,
                prompt_tokens=api_response.usage.prompt_tokens,
                total_tokens=api_response.usage.total_tokens
            )
        
        # Extract thinking blocks
        thinking_blocks = self._extract_thinking_blocks(message)

        # Extract reasoning content
        reasoning_content = None
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            reasoning_content = message.reasoning_content
            

        return Response(
            call_id=api_response.id,
            agent_name=self.agent_name,
            role=choice.message.role,
            content=content,
            tool_calls=[tc.model_dump() for tc in tool_calls],
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
            usage=usage
        )
    
    def _extract_thinking_blocks(self, message: Any) -> List[ThinkingBlock]:
        """Extract thinking blocks from message."""
        thinking_blocks = []
        
        if hasattr(message, 'thinking_blocks') and message.thinking_blocks:
            for block in message.thinking_blocks:
                if block.get('type') == 'thinking':
                    thinking_blocks.append(
                        ThinkingBlock(
                            type="thinking",
                            thinking=block.get('thinking', ''),
                            signature=block.get('signature')
                        )
                    )
                elif block.get('type') == 'redacted_thinking':
                    thinking_blocks.append(
                        ThinkingBlock(
                            type="redacted_thinking",
                            data=block.get('data', '')
                        )
                    )
        
        return thinking_blocks
    
    def _process_stream_chunk(
        self, 
        chunk: Any, 
        stream_state: 'StreamingState'
    ) -> Optional[StreamResponse]:
        """
        Process a streaming chunk.
        
        Args:
            chunk: Stream chunk from API
            stream_state: Current streaming state
            
        Returns:
            StreamResponse if there's content to yield
        """
        if not chunk.choices:
            return None
                
        delta = chunk.choices[0].delta
        content = None
        
        # Update role if present
        if hasattr(delta, 'role') and delta.role:
            stream_state.role = delta.role
        
        # Process content
        if hasattr(delta, 'content') and delta.content:
            stream_state.current_content += delta.content
            content = delta.content
        
        # Process tool calls
        if hasattr(delta, 'tool_calls') and delta.tool_calls:
            content = self._process_stream_tool_calls(delta.tool_calls, stream_state)
        
        # Process thinking content
        thinking_blocks = []
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
            stream_state.reasoning_content += delta.reasoning_content
        
        if hasattr(delta, 'thinking_blocks') and delta.thinking_blocks:
            thinking_blocks = self._extract_thinking_blocks(delta)
            stream_state.thinking_blocks.extend(thinking_blocks)
            for block in thinking_blocks:
                if block.type == "thinking":
                    stream_state.signature = block.signature
        
        # Extract usage if present
        usage = None
        if hasattr(chunk, 'usage'):
            usage = Usage(
                completion_tokens=chunk.usage.completion_tokens,
                prompt_tokens=chunk.usage.prompt_tokens,
                total_tokens=chunk.usage.total_tokens
            )
            stream_state.usage = usage
        
        # Create response if there's content
        if content or thinking_blocks or usage:
            return StreamResponse(
                call_id=chunk.id,
                agent_name=self.agent_name,
                role=stream_state.role,
                content=content,
                reasoning_content=delta.reasoning_content if hasattr(delta, 'reasoning_content') else None,
                is_thinking=not bool(stream_state.signature),
                is_tool_call=bool(stream_state.current_tool_id),
                tool_name=stream_state.current_tool_name,
                tool_id=stream_state.current_tool_id,
                thinking_blocks=thinking_blocks if thinking_blocks else None,
                usage=usage
            )
        
        return None
    
    def _process_stream_tool_calls(
        self, 
        tool_calls: List[Any], 
        stream_state: 'StreamingState'
    ) -> Optional[str]:
        """Process tool calls in stream chunks."""
        if not tool_calls:
            return None
        
        tc = tool_calls[0] #when streaming, there is only one tool call at each chunk
        
        # New tool call
        if tc.id is not None:
            stream_state.current_tool_id = tc.id
            stream_state.current_tool_name = tc.function.name
            stream_state.tool_calls_info[tc.id] = {
                "name": tc.function.name,
                "arguments": ""
            }
        
        # Accumulate arguments
        if hasattr(tc.function, 'arguments') and tc.function.arguments:
            if stream_state.current_tool_id:
                stream_state.tool_calls_info[stream_state.current_tool_id]["arguments"] += tc.function.arguments
            return tc.function.arguments
        
        return None
    
    async def _handle_response_content(self, response: Response) -> None:
        """
        Handle response content and tool calls.
        
        Args:
            response: The response to process
        """
        # Add content to conversation
        if response.content or response.thinking_blocks:
            content_objects = self._create_content_objects(
                response.content, 
                response.thinking_blocks
            )
            self._add_message(
                role="assistant",
                content=content_objects,
                usage=response.usage
            )
        
        # Process tool calls
        if response.tool_calls:
            await self._process_tool_calls(response.tool_calls)
    
    async def _handle_stream_completion(self, stream_state: 'StreamingState') -> None:
        """
        Handle completion of streaming.
        
        Args:
            stream_state: Final streaming state
        """
        # Consolidate content
        if stream_state.current_content or stream_state.thinking_blocks:
            content_objects = self._create_content_objects(
                stream_state.current_content,
                self._consolidate_thinking_blocks(stream_state.thinking_blocks)
            )
            self._add_message(
                role="assistant",
                content=content_objects,
                usage=stream_state.usage
            )
        
        # Process accumulated tool calls
        if stream_state.tool_calls_info:
            tool_calls = self._create_tool_calls_from_stream(stream_state.tool_calls_info)
            await self._process_tool_calls(tool_calls)
    
    def _create_content_objects(
        self, 
        text_content: str, 
        thinking_blocks: List[ThinkingBlock]
    ) -> List[ContentUnion]:
        """Create content objects from text and thinking blocks."""
        content_objects = []
        
        # Add thinking blocks first
        for block in thinking_blocks:
            if block.type == "thinking":
                content_objects.append(
                    ContentThinking(
                        type="thinking",
                        thinking=block.thinking,
                        signature=block.signature
                    )
                )
            elif block.type == "redacted_thinking":
                content_objects.append(
                    ContentRedactedThinking(
                        type="redacted_thinking",
                        data=block.data
                    )
                )
        
        # Add text content
        if text_content:
            content_objects.append(Content(type="text", text=text_content))
        
        return content_objects
    
    def _consolidate_thinking_blocks(
        self, 
        blocks: List[ThinkingBlock]
    ) -> List[ThinkingBlock]:
        """Consolidate thinking blocks by type."""
        consolidated = {}
        
        for block in blocks:
            if block.type == "thinking":
                if "thinking" not in consolidated:
                    consolidated["thinking"] = ThinkingBlock(
                        type="thinking",
                        thinking="",
                        signature=block.signature
                    )
                consolidated["thinking"].thinking += block.thinking
                if block.signature:
                    consolidated["thinking"].signature = block.signature
            elif block.type == "redacted_thinking":
                consolidated["redacted_thinking"] = block
        
        return list(consolidated.values())
    
    def _create_tool_calls_from_stream(
        self, 
        tool_calls_info: Dict[str, Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Create tool calls from streaming info."""
        return [
            {
                "index": i,
                "id": tool_id,
                "function": {
                    "name": info["name"],
                    "arguments": info["arguments"]
                },
                "type": "function"
            }
            for i, (tool_id, info) in enumerate(tool_calls_info.items())
        ]
    
    async def _process_tool_calls(self, tool_calls: List[Any]) -> None:
        """
        Process tool calls from response.
        
        Args:
            tool_calls: List of tool calls to process
        """
        # Validate and add tool calls to memory
        for tool_call in tool_calls:
            tool_dict = tool_call if isinstance(tool_call, dict) else tool_call.model_dump()
            
            # Check for task_completed
            if tool_dict["function"]["name"] == "task_completed":
                self._mark_task_complete = True
            
            self.tools_mem.append(tool_dict)
        
        # Add to conversation history
        self._add_message(role="assistant", tool_calls=self.tools_mem)
        
        # Execute tools
        await self._execute_function_tools()
        
        # Enqueue agent tools
        self._enqueue_agent_tools()

        # Handle task completion
        if hasattr(self, '_mark_task_complete') and self._mark_task_complete:
            self.state = AgentState.COMPLETE
            if self in self.agents_queue:
                self.agents_queue.remove(self)
            del self._mark_task_complete
        
    
    async def _execute_function_tools(self) -> None:
        """Execute function tools in parallel."""
        if not self.tools_mem[:]:
            return
        
        # Separate function tools from agent tools
        function_tasks = []

        for tool in self.tools_mem:
            tool_name = tool["function"]["name"]
            
            if tool_name in self.tools_functions:
                func = self.tools_functions[tool_name]
                task = self._create_tool_execution_task(tool, func)
                function_tasks.append(task)
        
        # Execute all function tools in parallel
        if function_tasks:
            try:
                await asyncio.gather(*function_tasks, return_exceptions=False)
            except Exception as e:
                logger.error(f"Error executing function tools: {e}")
                raise
    
    async def _create_tool_execution_task(
        self, 
        tool: Dict[str, Any], 
        func: Callable
    ) -> None:
        """Create an execution task for a tool."""
        try:
            await self.execute_func_tool(tool, func)
        except Exception as e:
            logger.error(f"Tool execution failed for {tool['function']['name']}: {e}")
            # Re-raise to be caught by gather
            raise
    
    def _enqueue_agent_tools(self) -> None:
        """Enqueue agent tools for execution."""
        for tool in self.tools_mem[:]:  # Use slice to avoid modification during iteration
            tool_name = tool["function"]["name"]
            
            if tool_name in self.tools_agent:
                try:
                    self._enqueue_single_agent_tool(tool)
                except Exception as e:
                    logger.error(f"Failed to enqueue agent tool {tool_name}: {e}")
                    self._handle_tool_error_sync(
                        f"Failed to create agent: {str(e)}",
                        tool,
                        e
                    )
    
    
    
    def _enqueue_single_agent_tool(self, tool: Dict[str, Any]) -> None:
        """Enqueue a single agent tool."""
        tool_name = tool["function"]["name"]
        agent_method = self.tools_agent[tool_name]
        
        # Parse arguments
        try:
            arguments = json.loads(tool["function"]["arguments"])
        except json.JSONDecodeError as e:
            raise ToolExecutionError(
                f"Invalid JSON arguments for tool {tool_name}: {tool['function']['arguments']}"
            ) from e
        
        # Create agent instance
        try:
            agent_instance = agent_method(self, **arguments)
            #check if is BaseTaskAgent
        except Exception as e:
            raise ToolExecutionError(
                f"Failed to create agent {tool_name}: {str(e)}"
            ) from e

        if not isinstance(agent_instance, BaseTaskAgent):
            raise ToolExecutionError(
                f"Tool {tool_name} is not a task agent"
            )

        # Validate streaming settings
        parent_stream = self.completion_kwargs.get("stream", False)
        child_stream = agent_instance.completion_kwargs.get("stream", False)
        
        if parent_stream is False and child_stream is True:
            raise StreamingMismatchError(
                f"Parent agent '{self.agent_name}' has stream=False, but child agent "
                f"'{agent_instance.agent_name}' has stream=True. When parent agent "
                f"is not streaming, child agents must also have stream=False."
            )


        # Configure agent instance
        agent_instance.agents_queue = self.agents_queue
        agent_instance.tool_call_id = tool["id"]
        agent_instance.tool_name = tool_name
        agent_instance.parent_agent = self
        
        # Add to hierarchy
        self.child_agents.append(agent_instance)
        self.agents_queue.appendleft(agent_instance)
        self.agents_queue_history.append(agent_instance)
        
        logger.debug(f"Enqueued agent tool: {tool_name}")    
    
    
    
    async def execute_func_tool(self, tool: Dict[str, Any], func: Callable) -> None:
        """
        Execute a single function tool.
        
        Args:
            tool: Tool call information
            func: Function to execute
        """
        tool_name = tool["function"]["name"]
        tool_id = tool["id"]
        
        if not self.silent:
            print(f"Executing tool: {tool_name} (agent: {self.agent_name})")
            logger.info(f"Executing tool: {tool_name} (agent: {self.agent_name})")
        
        try:
            # Parse arguments
            arguments = json.loads(tool["function"]["arguments"])
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(self, **arguments)
            else:
                # Run sync function in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, func, self, **arguments)
            
            # Process result
            await self._handle_tool_result(tool, result)
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON arguments for tool {tool_name}"
            logger.error(f"{error_msg}: {e}")
            await self._handle_tool_error(error_msg, tool, e)
            
        except TypeError as e:
            error_msg = f"Invalid arguments for tool {tool_name}"
            logger.error(f"{error_msg}: {e}")
            await self._handle_tool_error(error_msg, tool, e)
            
        except ToolExecutionError as e:
            # Tool execution errors are already logged by the decorator
            await self._handle_tool_error(str(e), tool, e)
            
        except Exception as e:
            # For unexpected errors, log with full traceback
            error_msg = f"Unexpected error in tool {tool_name}"
            logger.error(f"{error_msg}: {e}", exc_info=True)
            await self._handle_tool_error(error_msg, tool, e)





    async def _handle_tool_result(
        self, 
        tool: Dict[str, Any], 
        result: Any
    ) -> None:
        """
        Handle successful tool execution result.
        
        Args:
            tool: Tool call information
            result: Tool execution result
        """
        tool_name = tool["function"]["name"]
        tool_id = tool["id"]
        
        # Convert result to JSON
        try:
            result_json = json.dumps(result)
        except (TypeError, ValueError):
            # Fallback for non-JSON serializable results
            result_json = json.dumps({"result": str(result)})
        
        # Add result to conversation
        self._add_message(
            role="tool",
            content=result_json,
            tool_call_id=tool_id
        )
        
        # Handle task_completed specially
        if tool_name == "task_completed":
            await self._handle_task_completed(tool, result)
        else:
            await self._handle_regular_tool_completion(tool, result)
        
        # Remove tool from memory
        self.tools_mem = [t for t in self.tools_mem if t["id"] != tool_id]
    
    async def _handle_task_completed(
        self, 
        tool: Dict[str, Any], 
        result: Any
    ) -> None:
        """Handle completion of a task agent."""
        if not self.parent_agent:
            logger.warning("Task completion called but no parent agent exists")
            return
        
        # Add result to parent's conversation
        self.parent_agent._add_message(
            role="tool",
            content=json.dumps(result),
            tool_call_id=self.tool_call_id
        )
        
        # Handle next tool chaining
        parent_tool_method = self.parent_agent.tools_agent.get(self.tool_name)
        if parent_tool_method:
            next_tool = getattr(parent_tool_method, "next_tool", None)
            manual_call = getattr(parent_tool_method, "manual_call", None)
            
            if next_tool:
                self._setup_next_tool(
                    self.parent_agent,
                    next_tool,
                    manual_call,
                    result
                )
        
        # self._cleanup_tool_map(self.parent_agent,self.tool_name)
        
        # Remove this tool from the parent's tool memory
        self.parent_agent.tools_mem = [
            t for t in self.parent_agent.tools_mem 
            if t["id"] != self.tool_call_id
        ]
        
        # Mark as complete
        self.completed_task = True


    async def _handle_regular_tool_completion(
        self, 
        tool: Dict[str, Any], 
        result: Any
    ) -> None:
        """Handle completion of a regular function tool."""

        tool_name = tool["function"]["name"]
        tool_method = self.tools_functions.get(tool_name)

        if tool_method:
            next_tool = getattr(tool_method, "next_tool", None)
            manual_call = getattr(tool_method, "manual_call", None) # this is the function to be called to create a manual tool call
            if next_tool:
                self._setup_next_tool(
                    self,
                    next_tool,
                    manual_call,
                    result
                )

    def _setup_next_tool(
        self,
        agent: 'BaseAgent',
        next_tool: str,
        manual_call: Optional[Callable],
        result: Any
    ) -> None:
        """Set up the next tool in the chain."""
        agent.next_tool_forced.append({"tool_name": next_tool,"manual_call_args": None})
        try:
            if manual_call:
                agent.next_tool_forced[-1]["manual_call_args"] = manual_call(result)
            else:
                agent.next_tool_forced[-1]["manual_call_args"] = None
        except Exception as e:
            logger.error(f"Failed to create manual tool call: {e}")


    def _create_manual_tool_call(
        self,
        agent: 'BaseAgent',
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create a manual tool call and return it."""
        # Validate arguments against tool signature
        if tool_name in agent.tools_functions:
            tool_func = agent.tools_functions[tool_name]
        elif tool_name in agent.tools_agent:
            tool_func = agent.tools_agent[tool_name]
        else:
            logger.warning(f"Tool {tool_name} not found for manual call")
            return None
        
        # Check if the arguments are valid for the tool
        import inspect
        sig = inspect.signature(tool_func)
        expected_params = set(sig.parameters.keys()) - {'self'}
        provided_params = set(arguments.keys())
        
        if expected_params != provided_params:
            logger.warning(
                f"Manual call arguments mismatch for {tool_name}. "
                f"Expected: {expected_params}, Provided: {provided_params}"
            )
            return None
        
        # Create tool call
        manual_tool_call = {
            "id": f"manual_{self._generate_tool_id()}",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(arguments)
            },
            "type": "function"
        }
        
        # Return the tool call so it can be added to tools_mem
        return manual_tool_call
        # self._add_message(
        #     role="assistant",
        #     content=json.dumps(manual_tool_call),
        #     tool_call_id=manual_tool_call["id"]
        # )
        

    def _generate_tool_id(self) -> str:
        """Generate a unique tool ID."""
        import secrets
        import string
        
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(24))
    
    async def _handle_tool_error(
        self,
        error_message: str,
        tool: Dict[str, Any],
        exception: Exception
    ) -> None:
        """
        Handle tool execution errors.
        
        Args:
            error_message: User-friendly error message
            tool: Tool that failed
            exception: The exception that was raised
        """
        tool_name = tool["function"]["name"]
        tool_id = tool["id"]
        
        # Create error details
        error_details = {
            "error": error_message,
            "tool_name": tool_name,
            "exception_type": type(exception).__name__,
            "exception_message": str(exception)
        }
        
        # Add traceback only in verbose mode
        if self.error_verbosity == "verbose":
            error_details["traceback"] = traceback.format_exc()
        
        # Print errors based on verbosity setting
        if not self.silent:
            if self.error_verbosity == "minimal":
                print(f"Error in {tool_name}")
            elif self.error_verbosity == "normal":
                print(f"Tool error in {tool_name}: {str(exception)}")
            else:  # verbose
                print(f"Tool error in {tool_name}: {error_message}")
        
        # Add error message to conversation
        self._add_message(
            role="tool",
            content=json.dumps(error_details),
            tool_call_id=tool_id
        )
        
        # Remove failed tool from memory
        self.tools_mem = [t for t in self.tools_mem if t["id"] != tool_id]
        
        # Update retry count
        self.retry_count += 1
        
        # Check for critical errors that should stop execution
        is_critical_error = self._is_critical_tool_error(exception, tool_name)
        
        if is_critical_error:
            # For critical errors, transition agent to waiting state to prevent infinite loops
            if not self.silent:
                print(f"Critical error in {tool_name}. Agent execution stopped.")
            
            # For conversational agents, wait for user input
            if self.is_conversational:
                self.state = AgentState.WAITING_FOR_USER
            # For task agents, mark as complete if it's a creation error
            elif isinstance(self, BaseTaskAgent):
                self.state = AgentState.COMPLETE
                # Remove from queue to stop execution
                if self in self.agents_queue:
                    self.agents_queue.remove(self)
            else:
                self.state = AgentState.WAITING_FOR_USER
    
    def _is_critical_tool_error(self, exception: Exception, tool_name: str) -> bool:
        """
        Determine if a tool error is critical enough to stop agent execution.
        
        Args:
            exception: The exception that was raised
            tool_name: Name of the tool that failed
            
        Returns:
            True if this is a critical error that should stop execution
        """
        # Agent creation errors are always critical
        if "Failed to create agent" in str(exception):
            return True
        
        # BaseTaskAgent initialization errors (missing task_completed)
        if isinstance(exception, TypeError) and "task_completed" in str(exception):
            return True
        
        # ToolExecutionError from agent tools that failed during creation
        if isinstance(exception, ToolExecutionError) and tool_name in self.tools_agent:
            # Check if it's an agent creation failure
            exception_msg = str(exception).lower()
            if any(keyword in exception_msg for keyword in [
                "failed to create agent",
                "must implement 'task_completed'",
                "initialization failed",
                "agent initialization"
            ]):
                return True
        
        # Streaming mismatch errors
        if "StreamingMismatchError" in str(type(exception)):
            return True
        
        # Invalid tool schema errors
        if isinstance(exception, (TypeError, ValueError)) and "schema" in str(exception).lower():
            return True
        
        # If retry count is getting too high, consider it critical
        if self.retry_count >= (self.max_retries // 2):  # Half of max retries
            return True
        
        return False
    
    def _handle_tool_error_sync(
        self,
        error_message: str,
        tool: Dict[str, Any],
        exception: Exception
    ) -> None:
        """
        Synchronous version of _handle_tool_error for use in non-async contexts.
        
        Args:
            error_message: User-friendly error message
            tool: Tool that failed
            exception: The exception that was raised
        """
        tool_name = tool["function"]["name"]
        tool_id = tool["id"]
        
        # Create error details
        error_details = {
            "error": error_message,
            "tool_name": tool_name,
            "exception_type": type(exception).__name__,
            "exception_message": str(exception)
        }
        
        # Add traceback only in verbose mode
        if self.error_verbosity == "verbose":
            error_details["traceback"] = traceback.format_exc()
        
        # Print errors based on verbosity setting
        if not self.silent:
            if self.error_verbosity == "minimal":
                print(f"Error in {tool_name}")
            elif self.error_verbosity == "normal":
                print(f"Tool error in {tool_name}: {str(exception)}")
            else:  # verbose
                print(f"Tool error in {tool_name}: {error_message}")
        
        # Add error message to conversation
        self._add_message(
            role="tool",
            content=json.dumps(error_details),
            tool_call_id=tool_id
        )
        
        # Remove failed tool from memory
        self.tools_mem = [t for t in self.tools_mem if t["id"] != tool_id]
        
        # Update retry count
        self.retry_count += 1
        
        # Check for critical errors that should stop execution
        is_critical_error = self._is_critical_tool_error(exception, tool_name)
        
        if is_critical_error:
            # For critical errors, transition agent to waiting state to prevent infinite loops
            if not self.silent:
                print(f"Critical error in {tool_name}. Agent execution stopped.")
            
            # For conversational agents, wait for user input
            if self.is_conversational:
                self.state = AgentState.WAITING_FOR_USER
            # For task agents, mark as complete if it's a creation error
            elif isinstance(self, BaseTaskAgent):
                self.state = AgentState.COMPLETE
                # Remove from queue to stop execution
                if self in self.agents_queue:
                    self.agents_queue.remove(self)
            else:
                self.state = AgentState.WAITING_FOR_USER
    
    def add_message(
        self,
        role: str,
        content: Union[str, List[ContentUnion]],
        **kwargs
    ) -> None:
        """
        Add a message to the current agent's conversation.
        
        Args:
            role: Message role (user/assistant/system/tool)
            content: Message content (string or list of content objects)
            **kwargs: Additional message parameters
        """
        if not self.agents_queue:
            raise AgentExecutionError("No active agent in queue")
        
        current_agent = self.agents_queue[0]
        current_agent._add_message(role, content, **kwargs)
        
        # Update agent state when user provides input
        if role == "user" and current_agent.state == AgentState.WAITING_FOR_USER:
            current_agent.state = AgentState.READY
    
    def _add_message(
        self,
        role: str,
        content: Optional[Union[str, List[ContentUnion]]] = None,
        **kwargs
    ) -> None:
        """
        Internal method to add a message to this agent's conversation.
        
        Args:
            role: Message role
            content: Message content
            **kwargs: Additional message parameters
        """
        # Normalize content
        if isinstance(content, str):
            content_objects = [Content(type="text", text=content)]
        elif content is None:
            content_objects = []
        else:
            content_objects = content
        
        # Create and add message
        message = Message(
            role=role,
            agent_name=self.agent_name,
            content=content_objects,
            **kwargs
        )
        
        self.conv_history.messages.append(message)
        
        # Log message addition
        logger.debug(
            f"Added {role} message to {self.agent_name} "
            f"(length: {len(content_objects) if content_objects else 0})"
        )
    
    def get_conversation_history(
        self,
        agent_name: Optional[str] = None,
        include_hidden: bool = False,
        format: str = "default"
    ) -> Union[List[Message], List[Dict[str, Any]], str]:
        """
        Get conversation history with various formatting options.
        
        Args:
            agent_name: Filter by agent name
            include_hidden: Include hidden messages
            format: Output format ('default', 'openai', 'string')
            
        Returns:
            Formatted conversation history
        """
        messages = self.conv_history.get_messages(
            agent_name=agent_name or self.agent_name,
            include_hidden=include_hidden
        )
        
        if format == "default":
            return messages
        elif format == "openai":
            return [msg.to_oai_style() for msg in messages]
        elif format == "string":
            return self._format_messages_as_string(messages)
        else:
            raise ValueError(f"Invalid format: {format}")
    
    def _format_messages_as_string(self, messages: List[Message]) -> str:
        """Format messages as a readable string."""
        lines = []
        
        for msg in messages:
            role_prefix = f"[{msg.role.upper()}]"
            
            if msg.content:
                for content in msg.content:
                    if content.type == "text":
                        lines.append(f"{role_prefix} {content.text}")
                    elif content.type == "thinking":
                        lines.append(f"{role_prefix} [THINKING] {content.text}")
            
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    lines.append(
                        f"{role_prefix} [TOOL CALL] {tc['function']['name']}({tc['function']['arguments']})"
                    )
        
        return "\n".join(lines)


class StreamingState:
    """State management for streaming responses."""
    
    def __init__(self):
        self.role: str = "assistant"
        self.current_content: str = ""
        self.reasoning_content: str = ""
        self.current_tool_id: Optional[str] = None
        self.current_tool_name: Optional[str] = None
        self.tool_calls_info: Dict[str, Dict[str, str]] = {}
        self.thinking_blocks: List[ThinkingBlock] = []
        self.signature: Optional[str] = None
        self.usage: Optional[Usage] = None


class BaseTaskAgent(BaseAgent):
    """
    Base class for task-oriented agents.
    
    Task agents are designed to complete specific tasks and return
    results to their parent agents.
    """
    
    tool_call_id: Optional[str] = Field(None, description="ID of the tool call that created this agent")
    tool_name: Optional[str] = Field(None, description="Name of the tool that created this agent")
    completed_task: bool = Field(False, description="Whether the task has been completed")
    
    
    def model_post_init(self, __context) -> None:
        """Initialize task agent and verify task_completed implementation."""
        # Call parent initialization first
        super().model_post_init(__context)
        
        # Check if task_completed is implemented
        if not self._is_task_completed_implemented():
            raise TypeError(
                f"\n{self.__class__.__name__} must implement 'task_completed' method.\n\n"
                "Example:\n"
                "    @function_tool\n"
                "    def task_completed(self, result: str) -> Dict[str, Any]:\n"
                "        '''Complete the task and return results.\n"
                "        \n"
                "        Args:\n"
                "            result: The task result to return\n"
                "        '''\n"
                "        return {'status': 'success', 'result': result}\n"
            )
    
    def _is_task_completed_implemented(self) -> bool:
        """Check if task_completed is properly implemented."""
        # Get the task_completed method
        task_completed = getattr(self.__class__, 'task_completed', None)
        
        if not task_completed:
            return False
        
        # Check if it's decorated as a tool
        if not getattr(task_completed, 'is_tool', False):
            return False
        
        # Check if it's not the base implementation (if there is one)
        # Since there's no base implementation, we just need to check it exists and is a tool
        return True
    
    def get_task_status(self) -> Dict[str, Any]:
        """
        Get the current task status.
        
        Returns:
            Dictionary containing task status information
        """
        return {
            "agent_name": self.agent_name,
            "tool_name": self.tool_name,
            "completed": self.completed_task,
            "state": self.state.value,
            "parent": self.parent_agent.agent_name if self.parent_agent else None,
            "messages_count": len(self.conv_history.messages)
        }
    
    # @function_tool
    # def task_completed(self, result: str) -> Dict[str, Any]:
    #     """
    #     Complete the task and return results to parent.
        
    #     This method must be overridden by subclasses.
        
    #     Args:
    #         result: The result to return to the parent agent
            
    #     Returns:
    #         Dictionary containing the task result
    #     """
    #     raise NotImplementedError(
    #         "Subclasses must implement the task_completed method"
    #     )
    
