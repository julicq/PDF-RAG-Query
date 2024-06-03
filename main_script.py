import os
import shutil
from get_embedding_function import get_embedding_function
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from tqdm import tqdm
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

def add_to_chroma(chunks, use_tqdm=True):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    
    # Calculate chunk IDs
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Only add docs that are not in DB
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata.get("id") not in existing_ids]

    if new_chunks:
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]

        # Use tqdm for terminal progress bar if enabled
        if use_tqdm and len(new_chunks) > 0:
            chunk_iterator = tqdm(zip(new_chunks, new_chunk_ids), total=len(new_chunks), desc="Adding documents")
        else:
            chunk_iterator = zip(new_chunks, new_chunk_ids)
        
        for chunk, chunk_id in chunk_iterator:
            db.add_documents([chunk], ids=[chunk_id])
        
        # db.persist()
    else:
        print("No new documents to add")

def calculate_chunk_ids(chunks):
    # This will create ids
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
    # Ensure Chroma instance is closed
    try:
        db = Chroma(persist_directory=CHROMA_PATH)
        db.close()
    except Exception as e:
        print(f"Error closing the database: {e}")

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
