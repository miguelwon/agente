# agente/core/exceptions.py
"""Custom exceptions for the Agente framework."""


class AgenteError(Exception):
    """Base exception for all Agente errors."""
    pass


class AgentExecutionError(AgenteError):
    """Raised when agent execution fails."""
    pass


class ToolExecutionError(AgenteError):
    """Raised when a tool execution fails."""
    pass


class InvalidToolError(AgenteError):
    """Raised when a tool is invalid or not found."""
    pass


class MaxRetriesExceededError(AgenteError):
    """Raised when maximum retries are exceeded."""
    pass


class ValidationError(AgenteError):
    """Raised when validation fails."""
    pass

class StreamingMismatchError(AgenteError):
    """Raised when parent and child agents have incompatible streaming settings."""
    pass

