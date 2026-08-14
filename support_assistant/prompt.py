def build_prompt(question, documents):
    context = "\n\n".join(
        f"SOURCE: {document['filename']}\n{document['content']}"
        for document in documents
    )

    return f"""
You are a Zepto customer support assistant.

Answer the customer's question using only the provided support documents.
Do not invent information that is not present in the documents.
If the documents do not contain enough information to answer the question, say that the available support information does not provide the answer.

SUPPORT DOCUMENTS:
{context}

CUSTOMER QUESTION:
{question}

ANSWER:
"""