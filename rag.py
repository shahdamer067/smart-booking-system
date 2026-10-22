"""RAG chatbot: chunk the docs, embed them, search with FAISS, answer with Groq (Llama 3.1)."""
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_PATH = Path(__file__).parent / "data" / "booking_docs.txt"

PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant for an appointment booking service.
Use ONLY the context below to answer the question.
If the answer is not in the context, say you don't know.

CONTEXT:
{context}

QUESTION: {question}"""
)


def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def build_chain():
    """Build the retrieval chain. Reads GOOGLE_API_KEY and GROQ_API_KEY from the environment."""
    text = DOCS_PATH.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0.0)
    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
