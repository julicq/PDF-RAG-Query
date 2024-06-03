import os
import shutil
from get_embedding_function import get_embedding_function
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page id's
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    # Add or update docs
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    existing_hashes = {doc.metadata.get("hash") for doc in existing_items["documents"] if "hash" in doc.metadata}
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add docs that are not in DB
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["hash"] not in existing_hashes:
            new_chunks.append(chunk)
    
    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        
        batch_size = 100  # Adjust batch size as needed
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            new_chunk_ids = [chunk.metadata["id"] for chunk in batch]
            db.add_documents(batch, ids=new_chunk_ids)
        
        db.persist()
    else:
        print("No new documents to add")


def calculate_chunk_ids(chunks):
    # This will create id's
    # Page Source: Page Numbers : Chunk Index
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page id is the same as last one, increment index
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        # Calculate chunk id
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to page meta-data
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    from langchain_community.vectorstores import Chroma
    
    # Ensure Chroma instance is closed
    try:
        db = Chroma(persist_directory=CHROMA_PATH)
        db.close()
    except Exception as e:
        print(f"Error closing the database: {e}")

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    # Example usage
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
