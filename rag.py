import os

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# GET RETRIEVER
# ============================================================

def get_retriever():

    # --------------------------------------------------------
    # PDF DIRECTORY
    # --------------------------------------------------------

    pdf_directory = "documents"

    if not os.path.exists(pdf_directory):

        raise FileNotFoundError(
            f"Directory '{pdf_directory}' does not exist."
        )

    # --------------------------------------------------------
    # FIND PDF FILES
    # --------------------------------------------------------

    pdf_files = [
        file
        for file in os.listdir(pdf_directory)
        if file.lower().endswith(".pdf")
    ]

    if not pdf_files:

        raise FileNotFoundError(
            f"No PDF files found inside '{pdf_directory}'."
        )

    # --------------------------------------------------------
    # LOAD PDF DOCUMENTS
    # --------------------------------------------------------

    documents = []

    for pdf_file in pdf_files:

        pdf_path = os.path.join(
            pdf_directory,
            pdf_file
        )

        print(f"Loading: {pdf_file}")

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        documents.extend(docs)

    # --------------------------------------------------------
    # SPLIT DOCUMENTS
    # --------------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} document chunks."
    )

    # --------------------------------------------------------
    # OPENAI EMBEDDINGS
    # --------------------------------------------------------

    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
    )


    # --------------------------------------------------------
    # CREATE FAISS VECTOR STORE
    # --------------------------------------------------------

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # --------------------------------------------------------
    # CREATE RETRIEVER
    # --------------------------------------------------------

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4
        }
    )

    return retriever, len(chunks)
