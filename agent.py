from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph.message import AnyMessage, add_messages
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

load_dotenv()

simple_prompt = """
You are a helpful AI assistant. Please respond to the user's message in a friendly and informative way.
"""

# model = <your proxy model here>
model = ChatOpenAI(model="gpt-4o", temperature=0.7)


@tool
def add(a: int, b: int) -> int:
    """A tool to add two numbers together"""
    return a + b


model_with_tools = model.bind_tools([add])


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


tools_node = ToolNode([add])


async def llm_node(state: State) -> dict:
    result = await model_with_tools.ainvoke(
        [SystemMessage(content=simple_prompt)] + state["messages"]
    )
    return {"messages": [result]}


def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


builder = StateGraph(State)
builder.add_node("llm_node", llm_node)
builder.add_node("tools_node", tools_node)
builder.add_edge(START, "llm_node")
builder.add_conditional_edges(
    "llm_node",
    should_continue,
    {
        "continue": "tools_node",
        "end": END,
    },
)
builder.add_edge("tools_node", "llm_node")
builder.add_edge("llm_node", END)
graph = builder.compile()
