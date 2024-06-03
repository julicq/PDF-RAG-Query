import streamlit as st
import os
from main_script import load_documents, split_documents, add_to_chroma, clear_database
from query_script import query_rag

# Define paths
CHROMA_PATH = "chroma"
DATA_PATH = "data"

def save_uploaded_file(uploaded_file):
    # Ensure the DATA_PATH directory exists
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    # Save file with original name
    save_path = os.path.join(DATA_PATH, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
    return save_path

# Streamlit app
st.title("PDF Document Search with LangChain")

# Upload PDFs
st.header("Upload PDFs")
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = save_uploaded_file(uploaded_file)
        st.write(f"Uploaded {uploaded_file.name} to {save_path}")

    st.success("PDFs uploaded successfully!")

# Reset database
if st.button("Reset Database"):
    clear_database()
    st.success("Database reset successfully!")

# Add PDFs to the database
if st.button("Add PDFs to Database"):
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    st.success("PDFs added to the database successfully!")

# Query section
st.header("Query the PDF Database")
query_text = st.text_input("Enter your query:")
if st.button("Search"):
    if query_text:
        response_text = query_rag(query_text)
        st.write("Response:", response_text)
    else:
        st.error("Please enter a query.")
