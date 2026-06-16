from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.graph.state import GraphState
from src.graph.nodes import (
    query_rewriter,
    retriever_node,
    relevance_grader,
    web_fallback,
    answer_generator,
    hallucination_checker
)

# ─────────────────────────────────────────
# ROUTING FUNCTIONS
# These decide which node to go to next
# based on the current state
# ─────────────────────────────────────────

def route_after_grading(state: GraphState) -> str:
    """
    After relevance grading:
    - If chunks are relevant → go to answer generator
    - If not relevant → go to web fallback
    """
    if state["relevance_score"] == "relevant":
        print("Routing to: Answer Generator")
        return "answer_generator"
    else:
        print("Routing to: Web Fallback")
        return "web_fallback"


def route_after_hallucination_check(state: GraphState) -> str:
    """
    After hallucination check:
    - If grounded → END, show answer to user
    - If not grounded → loop back to answer generator
    """
    if state["hallucination_flag"] == "grounded":
        print("Routing to: END")
        return END
    else:
        print("Answer not grounded, regenerating...")
        return "answer_generator"


# ─────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────

def build_graph():
    """
    Builds and compiles the LangGraph graph.
    Returns a runnable graph with memory.
    """

    # Initialize the graph with our state
    graph = StateGraph(GraphState)

    # ── Add all nodes ──
    graph.add_node("query_rewriter", query_rewriter)
    graph.add_node("retriever_node", retriever_node)
    graph.add_node("relevance_grader", relevance_grader)
    graph.add_node("web_fallback", web_fallback)
    graph.add_node("answer_generator", answer_generator)
    graph.add_node("hallucination_checker", hallucination_checker)

    # ── Define the flow ──

    # Start at query rewriter
    graph.set_entry_point("query_rewriter")

    # query_rewriter → retriever
    graph.add_edge("query_rewriter", "retriever_node")

    # retriever → relevance grader
    graph.add_edge("retriever_node", "relevance_grader")

    # relevance grader → conditional split
    # relevant → answer_generator
    # not relevant → web_fallback
    graph.add_conditional_edges(
        "relevance_grader",
        route_after_grading,
        {
            "answer_generator": "answer_generator",
            "web_fallback": "web_fallback"
        }
    )

    # web_fallback → answer_generator
    graph.add_edge("web_fallback", "answer_generator")

    # answer_generator → hallucination_checker
    graph.add_edge("answer_generator", "hallucination_checker")

    # hallucination_checker → conditional split
    # grounded → END
    # not_grounded → answer_generator (loop back)
    graph.add_conditional_edges(
        "hallucination_checker",
        route_after_hallucination_check,
        {
            END: END,
            "answer_generator": "answer_generator"
        }
    )

    # ── Add memory ──
    memory = MemorySaver()

    # ── Compile ──
    app = graph.compile(checkpointer=memory)

    print("Graph built successfully!")
    return app