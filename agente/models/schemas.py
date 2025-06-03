"""Pydantic models for the Agente framework."""
from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class ContentType(str, Enum):
    """Enumeration of content types."""
    TEXT = "text"
    THINKING = "thinking"
    REDACTED_THINKING = "redacted_thinking"


class Usage(BaseModel):
    """Token usage information."""
    completion_tokens: int = Field(..., ge=0, description="Number of completion tokens")
    prompt_tokens: int = Field(..., ge=0, description="Number of prompt tokens")
    total_tokens: int = Field(..., ge=0, description="Total number of tokens")
    
    @validator('total_tokens')
    def validate_total(cls, v, values):
        """Ensure total equals sum of prompt and completion tokens."""
        expected = values.get('completion_tokens', 0) + values.get('prompt_tokens', 0)
        if v != expected:
            raise ValueError(f"Total tokens ({v}) doesn't match sum of parts ({expected})")
        return v


class FunctionCall(BaseModel):
    """Function call information."""
    arguments: str = Field(..., description="JSON-encoded function arguments")
    name: str = Field(..., min_length=1, description="Function name")
    
    @validator('arguments')
    def validate_json(cls, v):
        """Ensure arguments is valid JSON."""
        import json
        try:
            json.loads(v)
        except json.JSONDecodeError:
            raise ValueError("Arguments must be valid JSON")
        return v


class Content(BaseModel):
    """Text content."""
    type: Literal["text"] = Field("text", description="Content type")
    text: str = Field(..., description="Text content")


class ContentThinking(BaseModel):
    """Thinking content with optional signature."""
    type: Literal["thinking"] = Field("thinking", description="Content type")
    thinking: str = Field(..., description="Thinking text")
    signature: Optional[str] = Field(None, description="Optional signature")


class ContentRedactedThinking(BaseModel):
    """Redacted thinking content."""
    type: Literal["redacted_thinking"] = Field("redacted_thinking", description="Content type")
    data: str = Field(..., description="Redacted data")


# Union type for all content types
ContentUnion = Union[Content, ContentThinking, ContentRedactedThinking]


class ToolCall(BaseModel):
    """Tool call information."""
    index: Optional[int] = Field(None, ge=0, description="Tool call index")
    function: FunctionCall = Field(..., description="Function call details")
    id: str = Field(..., min_length=1, description="Unique tool call ID")
    type: Literal["function"] = Field("function", description="Tool call type")


class ThinkingBlock(BaseModel):
    """Thinking block information."""
    type: str = Field(..., description="Block type")
    thinking: Optional[str] = Field(None, description="Thinking content")
    signature: Optional[str] = Field(None, description="Signature")
    data: Optional[str] = Field(None, description="Additional data")
    
    @validator('type')
    def validate_type(cls, v):
        """Validate thinking block type."""
        if v not in ["thinking", "redacted_thinking"]:
            raise ValueError(f"Invalid thinking block type: {v}")
        return v


class Message(BaseModel):
    """Conversation message."""
    role: str = Field(..., description="Message role (system/user/assistant/tool)")
    agent_name: str = Field(..., min_length=1, description="Agent name")
    content: Optional[List[ContentUnion]] = Field(
        default_factory=list,
        description="Message content"
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls")
    tool_call_id: Optional[str] = Field(None, description="Tool call ID for responses")
    tool_name: Optional[str] = Field(None, description="Tool name for responses")
    hidden: bool = Field(False, description="Whether message is hidden")
    id: Optional[str] = Field(None, description="Message ID")
    usage: Optional[Usage] = Field(None, description="Token usage")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    
    @validator('role')
    def validate_role(cls, v):
        """Validate message role."""
        valid_roles = {"system", "user", "assistant", "tool"}
        if v not in valid_roles:
            raise ValueError(f"Invalid role: {v}. Must be one of {valid_roles}")
        return v
    
    def to_oai_style(self) -> Dict[str, Any]:
        """Convert to OpenAI API style format."""
        result = {"role": self.role}
        
        if self.content:
            # Convert content objects to appropriate format
            if len(self.content) == 1 and self.content[0].type == "text":
                result["content"] = self.content[0].text
            else:
                result["content"] = [c.model_dump() for c in self.content]
        
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
            
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
            
        return result
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Response(BaseModel):
    """Agent response."""
    call_id: str = Field(..., description="Unique call ID")
    agent_name: str = Field(..., description="Agent name")
    role: str = Field(..., description="Response role")
    content: str = Field("", description="Response content")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Tool calls")
    reasoning_content: Optional[str] = Field(None, description="Reasoning content")
    thinking_blocks: List[ThinkingBlock] = Field(
        default_factory=list,
        description="Thinking blocks"
    )
    usage: Optional[Usage] = Field(None, description="Token usage")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StreamResponse(BaseModel):
    """Streaming response chunk."""
    call_id: str = Field(..., description="Unique call ID")
    agent_name: str = Field(..., description="Agent name")
    role: str = Field(..., description="Response role")
    content: Optional[str] = Field(None, description="Response content chunk")
    reasoning_content: Optional[str] = Field(None, description="Reasoning content")
    is_thinking: bool = Field(False, description="Whether this is thinking content")
    is_tool_call: bool = Field(False, description="Whether this is a tool call")
    tool_name: Optional[str] = Field(None, description="Tool name if tool call")
    is_tool_exec: bool = Field(False, description="Whether tool is executing")
    tool_id: Optional[str] = Field(None, description="Tool call ID")
    thinking_blocks: Optional[List[ThinkingBlock]] = Field(None, description="Thinking blocks")
    usage: Optional[Usage] = Field(None, description="Token usage")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Chunk timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationHistory(BaseModel):
    """Conversation history container."""
    messages: List[Message] = Field(default_factory=list, description="List of messages")
    
    def add_message(self, **kwargs) -> Message:
        """Add a message to the history."""
        message = Message(**kwargs)
        self.messages.append(message)
        return message
    
    def get_messages(
        self,
        agent_name: Optional[str] = None,
        role: Optional[str] = None,
        include_hidden: bool = False
    ) -> List[Message]:
        """Get filtered messages."""
        messages = self.messages
        
        if not include_hidden:
            messages = [m for m in messages if not m.hidden]
            
        if agent_name:
            messages = [m for m in messages if m.agent_name == agent_name]
            
        if role:
            messages = [m for m in messages if m.role == role]
            
        return messages