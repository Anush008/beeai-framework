import asyncio

from dotenv import load_dotenv

from beeai_framework.agents.tool_calling import (
    ToolCallingAgent,
    ToolCallingAgentSuccessEvent,
)
from beeai_framework.backend import ChatModel
from beeai_framework.emitter import EventMeta
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool

load_dotenv()


def log_events(data: ToolCallingAgentSuccessEvent, event: EventMeta) -> None:
    step = data.state.steps[-1]
    print(
        f"🤖 {event.creator.meta.name}",  # type: ignore
        f"executed '{step.tool.name}' {'ability' if step.ability is not None else 'tool'}\n"  # type: ignore
        f" -> Input: {step.input}\n -> Output: {step.output}",
    )


async def main() -> None:
    agent = ToolCallingAgent(
        llm=ChatModel.from_name("ollama:granite3.2"),
        memory=UnconstrainedMemory(),
        tools=[DuckDuckGoSearchTool()],
        abilities=["reasoning"],
    )

    prompt = "How many continents are in the world?"
    response = await agent.run(prompt).on("success", log_events)
    print(response.result.text)


if __name__ == "__main__":
    asyncio.run(main())
