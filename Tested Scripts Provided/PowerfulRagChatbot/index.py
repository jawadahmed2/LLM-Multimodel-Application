from retriever import retriever
from langgraph.graph import END, StateGraph
from graph import GraphState, grade_documents, generate, transform_query, web_search, decide_to_generate, retrieve
import pprint
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    HumanMessage,
)
import json
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search)

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

# # Run
# inputs = {"question": "What are the types of agent memory?"}

# print(app.invoke(inputs))

events = app.stream(
    {
        "messages": [
            HumanMessage(
                content="What are the types of agent memory?"
            )
        ],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 150},
)
for s in events:
    print(s)
    print("----")

# for output in app.stream(inputs):
#     for key, value in output.items():
#         # Node
#         pprint.pprint(f"Node '{key}':")
#         # Optional: print full state at each node
#         # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
#     pprint.pprint("\n---\n")

# # Final generation
# pprint(value["generation"])

# for s in app.stream(
#     {"messages": [HumanMessage(content="Write a brief research report on pikas.")]},
#     {"recursion_limit": 100},
# ):
#     if "__end__" not in s:
#         print(s)
#         print("----")