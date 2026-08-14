from pathlib import Path


DOCS_DIR = Path(__file__).parent / "docs"


def load_documents():
    documents = []

    for path in sorted(DOCS_DIR.glob("*.txt")):
        # utf-8-sig automatically removes UTF-8 BOM if present
        content = path.read_text(encoding="utf-8-sig").strip()

        documents.append(
            {
                "filename": path.name,
                "content": content,
            }
        )

    return documents


if __name__ == "__main__":
    documents = load_documents()

    print(f"Loaded {len(documents)} documents.")

    for document in documents:
        print(f"- {document['filename']}")