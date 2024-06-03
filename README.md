# PDF Document Search with LangChain

This project implements a Streamlit app that allows users to upload PDF files, add them to a Chroma database, and query the database to retrieve information from the uploaded PDFs using LangChain.

## Features

- Upload PDF files through the Streamlit interface.
- Add uploaded PDFs to a Chroma vector store database.
- Reset the database to clear all stored documents.
- Query the database to retrieve information from the PDFs using LangChain.

## Requirements

- Python 3.7 or later
- The following Python packages:
  - streamlit
  - langchain
  - langchain_community
  - pypdf2
  - hashlib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pdf-document-search-langchain.git
   cd pdf-document-search-langchain

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required packages:
    ```bash
    pip install -r requirements.txt

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run streamlit_app.py

2. Upload PDFs:
    - Use the "Upload PDFs" section to upload multiple PDF files. The files will be saved in the `data` directory with their original names.

3. Add PDFs to the Database:
    - Click the "Add PDFs to Database" button to process and add the uploaded PDFs to the Chroma vector store database.

4. Reset Database:
    - Click the "Reset Database" button to clear all stored documents in the Chroma database (only works when database is not in use currently)

5. Query the Database:
    - Enter a query in the "Enter your query" text input and click the "Search" button to retrieve information from the PDFs in the database.

## Project Structure

- app.py: The main Streamlit app script.
- main_script.py: Contains functions to load documents, split them into chunks, and add them to the Chroma database.
- query_script.py: Contains functions to query the Chroma database using LangChain.
- get_embedding_function.py: Contains the function to get the embedding function for Chroma.
- data/: Directory where uploaded PDF files are stored.
- chroma/: Directory where the Chroma database is stored.

## Acknoledgements
    -[Langchain](https://github.com/langchain-ai/langchain)
    -[Streamlit](https://www.streamlit.io/)