import os
from typing import AsyncGenerator
import chromadb

from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, Docx2txtLoader
from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv

load_dotenv()

# ---------------- EMBEDDINGS ----------------
class CustomEmbedding:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


class EmbeddingsManager:
    _embeddings = None

    @classmethod
    def get_embeddings(cls):
        if cls._embeddings is None:
            cls._embeddings = CustomEmbedding()
        return cls._embeddings


# ---------------- AGENT ----------------
class DocumentQAAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-3.5-turbo"
        )

        self.embeddings = EmbeddingsManager.get_embeddings()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        self.chroma_client = chromadb.Client()
        self.collection_name = f"temp_{os.urandom(4).hex()}"
        self.vector_store = None


    # ---------------- LOAD DOC ----------------
    def _load_documents(self, file_path, file_type, file_name):
        if file_type == "pdf":
            loader = PyMuPDFLoader(file_path)
        elif file_type in ["txt", "text"]:
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_type in ["docx", "doc"]:
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file_name
            if "page" not in doc.metadata:
                doc.metadata["page"] = 1

        return docs


    # ---------------- PROCESS ----------------
    def process_document(self, file_path, file_type, file_name):
        docs = self._load_documents(file_path, file_type, file_name)
        chunks = self.text_splitter.split_documents(docs)

        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                client=self.chroma_client,
                collection_name=self.collection_name
            )
        else:
            self.vector_store.add_documents(chunks)

        return len(chunks)


    # ---------------- RETRIEVE ----------------
    def _retrieve_context(self, query):
        if not self.vector_store:
            return "No documents uploaded."

        docs = self.vector_store.similarity_search(query, k=5)

        context = "\n\n".join([d.page_content for d in docs])
        return context


    # ---------------- ANSWER ----------------
    async def answer_question(self, question: str) -> AsyncGenerator[str, None]:
        context = self._retrieve_context(question)

        if context == "No documents uploaded.":
            yield "Please upload documents first."
            return

        prompt = f"""
        Answer based ONLY on the context below:

        Context:
        {context}

        Question:
        {question}
        """

        try:
            response = self.llm.invoke(prompt).content
        except Exception as e:
            response = f"Error: {str(e)}"

        # stream
        for i in range(0, len(response), 20):
            yield response[i:i+20]


    async def run_document_agent_stream(self, question):
        async for token in self.answer_question(question):
            yield token


    def cleanup(self):
        if self.vector_store:
            self.chroma_client.delete_collection(self.collection_name)