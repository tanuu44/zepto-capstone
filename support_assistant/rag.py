from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import load_documents


class RAGRetriever:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name="zepto_support")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        self._build_index()

    def _build_index(self):
        documents = load_documents()

        if not documents:
            raise RuntimeError("No support documents found in docs/.")

        ids = [document["filename"] for document in documents]
        texts = [document["content"] for document in documents]

        embeddings = self.embedding_model.encode(texts).tolist()

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"filename": document["filename"]} for document in documents],
        )

    def search(self, question: str, top_k: int = 3):
        query_embedding = self.embedding_model.encode([question]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        return [
            {
                "filename": metadata.get("filename", "unknown"),
                "content": content,
            }
            for content, metadata in zip(documents, metadatas)
        ]


if __name__ == "__main__":
    retriever = RAGRetriever()
    results = retriever.search("How long do I have to report a damaged item?")

    for result in results:
        print(f"SOURCE: {result['filename']}")
        print(result["content"])
        print()
