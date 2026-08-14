import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from models import ChatResponse
from rag import RAGRetriever
from prompt import build_prompt


# ---------------------------------------------------------
# State
# ---------------------------------------------------------

class SupportState(TypedDict, total=False):
    query: str
    intent: str
    documents: list
    answer: str
    sources: list[str]
    confidence: float
    response: dict


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

def mock_llm_enabled() -> bool:
    return os.getenv("MOCK_LLM", "1") == "1"


# ---------------------------------------------------------
# Retriever
# ---------------------------------------------------------

retriever = RAGRetriever()


# ---------------------------------------------------------
# Intent Classification
# ---------------------------------------------------------

def classify_intent(state: SupportState) -> SupportState:
    query = state["query"].lower().strip()

    policy_keywords = [
        "delivery",
        "deliver",
        "delivery time",
        "return",
        "returns",
        "refund",
        "refundable",
        "membership",
        "tracking",
        "track",
        "cancel",
        "cancellation",
        "gift card",
        "giftcard",
        "support",
        "support hours",
        "damaged",
        "damage",
        "damaged item",
        "missing",
        "missing item",
        "item damaged",
        "item missing",
        "replacement",
        "replace",
        "order",
        "orders",
        "complaint",
    ]

    if any(keyword in query for keyword in policy_keywords):
        intent = "policy_question"
    else:
        intent = "general_question"

    return {
        **state,
        "intent": intent,
    }


# ---------------------------------------------------------
# Retrieve and Answer
# ---------------------------------------------------------

def retrieve_and_answer(state: SupportState) -> SupportState:
    query = state["query"]

    documents = retriever.search(
        query,
        top_k=3,
    )

    if not documents:
        response = ChatResponse(
            answer="No relevant Zepto support information was found.",
            sources=[],
            confidence=0.0,
        )

        return {
            **state,
            "documents": [],
            "answer": response.answer,
            "sources": [],
            "confidence": 0.0,
            "response": response.model_dump(),
        }

    # Build the prompt from retrieved documents.
    prompt = build_prompt(
        query,
        documents,
    )

    # -----------------------------------------------------
    # MOCK_LLM mode
    #
    # Since no real LLM provider is configured, return the
    # retrieved support content directly.
    # -----------------------------------------------------

    if mock_llm_enabled():
        context_parts = []

        for document in documents:
            filename = document.get("filename", "unknown")
            content = document.get("content", "").strip()

            if content:
                context_parts.append(
                    f"SOURCE: {filename}\n{content}"
                )

        answer = (
            "Based on the retrieved Zepto support information:\n\n"
            + "\n\n".join(context_parts)
        )

        sources = [
            document.get("filename", "unknown")
            for document in documents
        ]

        response = ChatResponse(
            answer=answer,
            sources=sources,
            confidence=0.9,
        )

        return {
            **state,
            "documents": documents,
            "answer": response.answer,
            "sources": response.sources,
            "confidence": response.confidence,
            "response": response.model_dump(),
        }

    # -----------------------------------------------------
    # Optional real LLM path
    # -----------------------------------------------------

    answer = (
        "Real LLM mode is enabled, but no LLM provider has "
        "been configured yet.\n\n"
        "Retrieved context:\n\n"
        + prompt
    )

    sources = [
        document.get("filename", "unknown")
        for document in documents
    ]

    response = ChatResponse(
        answer=answer,
        sources=sources,
        confidence=0.5,
    )

    return {
        **state,
        "documents": documents,
        "answer": response.answer,
        "sources": sources,
        "confidence": 0.5,
        "response": response.model_dump(),
    }


# ---------------------------------------------------------
# Direct Answer
# ---------------------------------------------------------

def direct_answer(state: SupportState) -> SupportState:
    response = ChatResponse(
        answer=(
            "I can only answer questions about Zepto policies "
            "right now."
        ),
        sources=[],
        confidence=1.0,
    )

    return {
        **state,
        "documents": [],
        "answer": response.answer,
        "sources": [],
        "confidence": 1.0,
        "response": response.model_dump(),
    }


# ---------------------------------------------------------
# Routing
# ---------------------------------------------------------

def route_after_classification(state: SupportState) -> str:
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# ---------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------

def build_graph():
    workflow = StateGraph(SupportState)

    workflow.add_node(
        "classify_intent",
        classify_intent,
    )

    workflow.add_node(
        "retrieve_and_answer",
        retrieve_and_answer,
    )

    workflow.add_node(
        "direct_answer",
        direct_answer,
    )

    workflow.add_edge(
        START,
        "classify_intent",
    )

    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )

    workflow.add_edge(
        "retrieve_and_answer",
        END,
    )

    workflow.add_edge(
        "direct_answer",
        END,
    )

    return workflow.compile()


# ---------------------------------------------------------
# Compiled Graph
# ---------------------------------------------------------

support_graph = build_graph()


# ---------------------------------------------------------
# Local Test
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("POLICY QUESTION")
    print("=" * 60)

    result = support_graph.invoke(
        {
            "query": (
                "How long do I have to report "
                "a damaged item?"
            )
        }
    )

    print(result["response"])

    print()
    print("=" * 60)
    print("DELIVERY QUESTION")
    print("=" * 60)

    result = support_graph.invoke(
        {
            "query": "What is the delivery time?"
        }
    )

    print(result["response"])

    print()
    print("=" * 60)
    print("GENERAL QUESTION")
    print("=" * 60)

    result = support_graph.invoke(
        {
            "query": "What is artificial intelligence?"
        }
    )

    print(result["response"])