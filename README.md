# agente


A very simple Python framework for building AI Agents. 

## Overview

Agente is a Python framework that allows you to create AI agents just like you create classes and methods.

The Multi-agent orchestration is supported in an hierarchical way, starting from a main agent that can delegate tasks to specialized agents. 


## Features

- Simple agent creation easily customizable
- Support for streaming responses
- Tool integration capabilities
- Multi-agent orchestration
- Autonomous agent tool  that allows an agent to create its own tools (experimental)

## Installation

```bash
pip install agente
```

## Quick Start

Here's a simple example of creating a conversational agent:

```python
from agente.core.base import BaseAgent
from dotenv import load_dotenv

# Load environment variables (requires OpenAI API key)
load_dotenv()

class SimpleAgent(BaseAgent):
    agent_name: str = "SimpleAgent"
    system_prompt: str = "You are a helpful AI assistant."
    completion_kwargs: dict = {
        "model": "gpt-4",
        "stream": False,
        "temperature": 1.0,
        "max_tokens": 500,
    }

# Create agent instance
agent = SimpleAgent()

# Add a message
agent.add_message("user", "Tell me a joke about programming.")

# Run the agent and get responses
responses = [r async for r in agent.run()]

# Print the last response
print(responses[-1].content)
```

## Advanced Usage

### Adding Tools

Agents can be enhanced with tools using the `@function_tool` decorator:

```python
from agente.core.decorators import function_tool

class ToolAgent(BaseAgent):
    agent_name: str = "ToolAgent"
    
    @function_tool
    async def calculate_sum(self, a: int, b: int) -> int:
        """Calculate the sum of two numbers."""
        return a + b
```

### Creating Multi-Agent Systems

You can create complex multi-agent systems where agents can interact with each other:

```python
from agente.core.base import BaseAgent,BaseTaskAgent
from agente.core.decorators import function_tool,agent_tool
import random
from dotenv import load_dotenv
load_dotenv()

class JokeTeller(BaseTaskAgent):
    agent_name: str = "JokeTeller"
    system_prompt:str = "Your task is to write a funny joke."
    completion_kwargs: dict = {
        "model": "gpt-4o-mini",
        "stream": False,
    }

    @function_tool
    def complete_task(self,joke:str):
        """To be used as a tool to complete the task.

        Args:
            joke: The joke to return.
        """
        return joke



class MainAgent(BaseAgent):
    agent_name: str = "main_agent"
    
    @function_tool(next_tool = "get_joke")
    def random_topic(self):
        """Tool to get a random topic.
        """
        topics = ["programming","science","animals","food","sports"]
        topic = random.choice(topics)

        return topic



    @agent_tool()
    def get_joke(self,joke_topic:str):
        """Tool to get a joke.

        Args:
            joke_topic: The topic of the joke.
        """

        joke_agent = JokeTeller()
        joke_agent.add_message("user", "Tell me a joke about "+joke_topic)
        return joke_agent
    
example_agent = MainAgent()
example_agent.add_message("user", "Tell me a joke.")
responses = [r async for r in example_agent.run()]
print(example_agent.conv_history.messages[-1].content)
```

## Examples

For more examples, check out the examples directory:

1. Simple Conversational Agent
2. Data Analysis Agent
3. Scientific Paper Research Agent
4. Autonomous Agent with Dynamic Tools

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
