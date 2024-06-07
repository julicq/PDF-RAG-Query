from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function
from langdetect import detect
from googletrans import Translator

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question} using Language you've been asked with
"""

def query_rag(query_text: str):
    # Prepare DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search DB
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Detect the language of the query text
    query_language = detect(query_text)

    # Translate the query to English if it's not already in English
    if query_language != 'en':
        translator = Translator()
        query_text_en = translator.translate(query_text, src=query_language, dest='en').text
    else:
        query_text_en = query_text

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text_en)

    model = Ollama(model="llama3:8b-instruct-q8_0")
    response_text = model.invoke(prompt)

    # Translate the response back to the original language if it was translated
    if query_language != 'en':
        translator = Translator()
        response_text = translator.translate(response_text, src='en', dest=query_language).text

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"{response_text}\nSources: {sources}"
    print(formatted_response)
    return formatted_response
