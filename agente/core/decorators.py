from functools import wraps
import asyncio
from typing import Callable, Union, Any

def function_tool(func: Callable) -> Callable:
    """
    Decorator to mark a method as a function tool.
    
    Args:
        func: The function to be decorated
        
    Returns:
        Decorated function with tool metadata
    """
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
    return wrapper

def agent_tool(func: Union[type, Callable]) -> Union[type, Callable]:
    """
    Decorator to mark a method as an agent tool.
    Must return an instance of BaseTaskAgent.
    
    Args:
        func: The function to be decorated
        
    Returns:
        Decorated function with agent tool metadata
    """
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
    return wrapper