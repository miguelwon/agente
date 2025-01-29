from functools import wraps
import asyncio
from typing import Callable, Union, Any, Dict


def function_tool(
    func: Union[Callable, None] = None,
    *,
    ignore: list[str] = None,
    next_tool: str = None,
    manual_call: Callable[[Any], Dict] = None
) -> Callable:
    """
    Decorator to mark a method as a function tool.
    
    Args:
        func: The function to be decorated
        ignore: List of parameter names to ignore
        
    Returns:
        Decorated function with tool metadata
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)
            except Exception as e:
                raise ValueError(f"Error in function tool {func.__name__}: {str(e)}") from e


        wrapper.is_tool = True
        wrapper.is_agent = False
        wrapper.ignored_params = ignore or []
        wrapper.next_tool = next_tool if next_tool else None
        wrapper.manual_call = manual_call if manual_call else None
        return wrapper

    if func is None:
        return decorator
    return decorator(func)



def agent_tool(
    func: Union[type, Callable] = None,
    *,
    next_tool: str = None,
    manual_call: Callable[[Any], Dict] = None
) -> Union[type, Callable]:
    """
    Decorator to mark a method as an agent tool.
    Must return an instance of BaseTaskAgent.
    
    Args:
        func: The function to be decorated
        next_tool: Name of the next agent to call after task completion
        manual_call: Callable that takes the complete_task output and returns parameters
                    for the next agent call
        
    Returns:
        Decorated function with agent tool metadata
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                agent_instance = func(self, *args, **kwargs)
                if not agent_instance.__class__.__bases__[0].__name__ == "BaseTaskAgent":
                    raise TypeError(
                        f"@agent_tool must return a BaseTaskAgent instance, "
                        f"got {type(agent_instance).__name__} instead"
                    )
                return agent_instance
            except Exception as e:
                raise ValueError(f"Error in agent tool {func.__name__}: {str(e)}") from e

        wrapper.is_tool = True
        wrapper.is_agent = True
        wrapper.next_tool = next_tool if next_tool else None
        wrapper.manual_call = manual_call if manual_call else None
        return wrapper

    if func is None:
        return decorator
    return decorator(func)

