# agente/core/decorators.py
"""Decorators for defining agent tools and functions."""
import asyncio
from functools import wraps
from typing import Callable, Union, Any, Dict, List, Optional, Type
import inspect
import logging

from .exceptions import ToolExecutionError, InvalidToolError

logger = logging.getLogger(__name__)


def function_tool(
    func: Optional[Callable] = None,
    *,
    ignore: Optional[List[str]] = None,
    next_tool: Optional[str] = None,
    manual_call: Optional[Callable[[Any], Dict]] = None,
    retry_on_error: bool = False,
    max_retries: int = 3
) -> Callable:
    """
    Decorator to mark a method as a function tool.
    
    This decorator transforms a regular method into a tool that can be called
    by the agent during execution.
    
    Args:
        func: The function to be decorated
        ignore: List of parameter names to ignore in the tool schema
        next_tool: Name of the tool to call next after this one completes
        manual_call: Function to transform the output before passing to next_tool
        retry_on_error: Whether to retry on error
        max_retries: Maximum number of retry attempts
        
    Returns:
        Decorated function with tool metadata
        
    Example:
        @function_tool
        def search_web(self, query: str) -> str:
            '''Search the web for information.
            
            Args:
                query: The search query
            '''
            return perform_search(query)
    """
    def decorator(func: Callable) -> Callable:
        # Validate function signature
        sig = inspect.signature(func)
        if 'self' not in sig.parameters:
            raise InvalidToolError(
                f"Tool function '{func.__name__}' must have 'self' as first parameter"
            )
        
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs) -> Any:
            """Async wrapper for tool execution."""
            retries = 0
            last_error = None
            
            while retries <= max_retries:
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(self, *args, **kwargs)
                    else:
                        result = func(self, *args, **kwargs)
                    
                    # Log successful execution
                    logger.debug(f"Tool '{func.__name__}' executed successfully")
                    return result
                    
                except Exception as e:
                    last_error = e
                    retries += 1
                    
                    if not retry_on_error or retries > max_retries:
                        error_msg = f"Error in tool '{func.__name__}': {str(e)}"
                        # Log at DEBUG level to reduce noise - the base agent will handle user-facing messages
                        logger.debug(error_msg, exc_info=True)
                        raise ToolExecutionError(error_msg) from e
                    
                    # For retries, log at DEBUG level
                    logger.debug(
                        f"Tool '{func.__name__}' failed (attempt {retries}/{max_retries}): {e}"
                    )
                    await asyncio.sleep(0.5 * retries)  # Exponential backoff
            
            raise ToolExecutionError(
                f"Tool '{func.__name__}' failed after {max_retries} retries"
            ) from last_error
        
        # Store metadata
        async_wrapper.is_tool = True
        async_wrapper.is_agent = False
        async_wrapper.ignored_params = ignore or []
        async_wrapper.next_tool = next_tool
        async_wrapper.manual_call = manual_call
        async_wrapper.retry_on_error = retry_on_error
        async_wrapper.max_retries = max_retries
        
        # Preserve original function for inspection
        async_wrapper.__wrapped__ = func
        
        return async_wrapper
    
    if func is None:
        return decorator
    return decorator(func)


def agent_tool(
    func: Optional[Union[Type, Callable]] = None,
    *,
    next_tool: Optional[str] = None,
    manual_call: Optional[Callable[[Any], Dict]] = None,
    timeout: Optional[float] = None
) -> Union[Type, Callable]:
    """
    Decorator to mark a method as an agent tool.
    
    This decorator marks a method that returns a BaseTaskAgent instance,
    allowing hierarchical agent composition.
    
    Args:
        func: The function to be decorated
        next_tool: Name of the next agent/tool to call after task completion
        manual_call: Function to transform task_completed output for next agent
        timeout: Maximum time in seconds to wait for agent completion
        
    Returns:
        Decorated function with agent tool metadata
        
    Example:
        @agent_tool(next_tool="summarize_results")
        def research_agent(self, topic: str) -> ResearchAgent:
            '''Create a research agent for the topic.
            
            Args:
                topic: The topic to research
            '''
            return ResearchAgent(
                agent_name=f"researcher_{topic}",
                topic=topic
            )
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            """Wrapper for agent tool execution."""
            try:
                # Execute the function - it should return an agent instance
                agent_instance = func(self, *args, **kwargs)
                
                # Validate return type
                if not hasattr(agent_instance, '__class__'):
                    raise TypeError(
                        f"Agent tool '{func.__name__}' must return an object instance"
                    )
                
                # Check if it's a BaseTaskAgent (more robust check)
                base_classes = [base.__name__ for base in agent_instance.__class__.__mro__]
                if "BaseTaskAgent" not in base_classes:
                    raise TypeError(
                        f"Agent tool '{func.__name__}' must return a BaseTaskAgent instance, "
                        f"got {type(agent_instance).__name__} instead"
                    )
                
                # Apply timeout if specified
                if timeout:
                    agent_instance.completion_kwargs["timeout"] = timeout
                
                logger.debug(f"Agent tool '{func.__name__}' created agent instance")
                return agent_instance
                
            except Exception as e:
                error_msg = f"Error in agent tool '{func.__name__}': {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise ToolExecutionError(error_msg) from e
        
        # Store metadata
        wrapper.is_tool = True
        wrapper.is_agent = True
        wrapper.next_tool = next_tool
        wrapper.manual_call = manual_call
        wrapper.timeout = timeout
        
        # Preserve original function
        wrapper.__wrapped__ = func
        
        return wrapper  # Return the synchronous wrapper, not async
    
    if func is None:
        return decorator
    return decorator(func)