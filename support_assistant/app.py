from fastapi import FastAPI
from models import ChatRequest, ChatResponse
from graph import support_graph


app = FastAPI(
    title="Zepto Support Assistant",
    version="1.0.0",
)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "Zepto Support Assistant",
    }


@app.post("/ask", response_model=ChatResponse)
def ask(request: ChatRequest):
    result = support_graph.invoke(
        {
            "query": request.query,
        }
    )

    return ChatResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0),
    )