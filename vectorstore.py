from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings import get_embeddings
from typing import List

def load_documents() -> List[Document]:
    loader = CSVLoader('dataset.csv')
    return loader.load()


def split_docs(docs: List[Document]) -> List[Document]:
    CHUNK_SIZE = 500
    OVERLAP_SIZE = 50
    

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP_SIZE
    )

    return splitter.split_documents(docs)


def embedding(docs: List[Document]):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )
    return vectorstore


def save_vector_to_local(vectorstore):
    path_str = './exp-faiss'    
    vectorstore.save_local(path_str)


def load_vector_from_local():
    path_str = './exp-faiss'
    return FAISS.load_local(
        path_str,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )

def init_vectorstore():
    docs = load_documents()
    # split_documents = split_docs(docs)
    vectorstore = embedding(docs)
    save_vector_to_local(vectorstore)
    return vectorstore