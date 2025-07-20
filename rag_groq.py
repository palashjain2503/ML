import os
import time
import shutil
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq  # ✅ Correct import for Groq

# Configuration
GROQ_API_KEY = "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"  # Replace with your Groq key
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

PDF_PATH = r"C:\Users\palas\Desktop\programs\project\utilities\RAG\IPC.pdf"  # Place your legal PDF in the same directory or give full path
VECTORSTORE_PATH = r"C:\Users\palas\Desktop\programs\project\utilities\RAG\rag_vectorstore"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

class RAGSystem:
    def __init__(self, pdf_path: str, vectorstore_path: str):
        self.pdf_path = pdf_path
        self.vectorstore_path = vectorstore_path
        self.vectorstore = None
        self.embeddings = None
        self.llm = ChatGroq(
            temperature=0.2,
            model_name="llama-3.1-8b-instant"
        )

    def setup_embeddings(self):
        if self.embeddings is None:
            print("🔍 Setting up embeddings...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-base-en-v1.5",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✅ Embeddings ready!")

    def vectorstore_exists(self) -> bool:
        return os.path.exists(self.vectorstore_path) and os.listdir(self.vectorstore_path)

    def create_vectorstore_from_pdf(self, force_rebuild: bool = False) -> bool:
        if self.vectorstore_exists() and not force_rebuild:
            print(f"📂 Vectorstore already exists at {self.vectorstore_path}")
            choice = input("Do you want to rebuild it? (y/N): ").lower()
            if choice != 'y':
                print("✅ Using existing vectorstore")
                return self.load_existing_vectorstore()


        if force_rebuild and self.vectorstore_exists():
            print("🗑️ Removing old vectorstore...")
            shutil.rmtree(self.vectorstore_path)
            time.sleep(1)

        print("🏗️ Building vectorstore from PDF...")
        chunks = self._load_and_chunk_pdf()
        if not chunks:
            return False

        self.setup_embeddings()
        return self._create_vectorstore(chunks)

    def _load_and_chunk_pdf(self) -> List[Document]:
        print(f"📖 Loading PDF: {self.pdf_path}")
        if not os.path.exists(self.pdf_path):
            print(f"❌ PDF file not found at: {self.pdf_path}")
            return []

        try:
            loader = PyPDFLoader(self.pdf_path)
            pages = loader.load()
            print(f"✅ Loaded {len(pages)} pages")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks = splitter.split_documents(pages)
            print(f"✅ Created {len(chunks)} chunks")
            return chunks
        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
            return []

    def _create_vectorstore(self, chunks: List[Document]) -> bool:
        print(f"💾 Creating persistent vectorstore at {self.vectorstore_path}...")
        try:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.vectorstore_path
            )
            print(f"✅ Vectorstore created with {len(chunks)} documents")
            return True
        except Exception as e:
            print(f"❌ ChromaDB creation failed: {e}")
            return self._create_vectorstore_batch(chunks)

    def _create_vectorstore_batch(self, chunks: List[Document]) -> bool:
        try:
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.vectorstore_path
            )
            batch_size = 25
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                print(f"📥 Adding batch {i // batch_size + 1}")
                self.vectorstore.add_documents(batch)
            self.vectorstore.persist()
            print("✅ Batch creation completed and persisted!")
            return True
        except Exception as e:
            print(f"❌ Batch creation failed: {e}")
            return False

    def load_existing_vectorstore(self) -> bool:
        if not self.vectorstore_exists():
            print(f"❌ No vectorstore found at {self.vectorstore_path}")
            return False

        try:
            print(f"📂 Loading existing vectorstore from {self.vectorstore_path}")
            self.setup_embeddings()
            self.vectorstore = Chroma(
                persist_directory=self.vectorstore_path,
                embedding_function=self.embeddings
            )
            count = len(self.vectorstore.get()['ids'])
            print(f"✅ Loaded vectorstore with {count} documents")
            return True
        except Exception as e:
            print(f"❌ Error loading vectorstore: {e}")
            return False

    def search_similar_documents(self, query: str) -> List[tuple]:
        if not self.vectorstore:
            return []
        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=TOP_K)
            sorted_docs = sorted(docs_with_scores, key=lambda x: x[1])
            print(f"🔍 Found {len(sorted_docs)} similar documents")
            return sorted_docs
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []

    def generate_answer(self, query: str, docs_with_scores: List[tuple]) -> str:
        if not docs_with_scores:
            return "No relevant documents found."

        context_parts = []
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            similarity_percent = max(0, (1 - score) * 100)
            context_parts.append(
                f"Document {i} (Similarity: {similarity_percent:.1f}%):\n{doc.page_content[:600]}"
            )

        context = "\n\n".join(context_parts)
        prompt = f"""You are a helpful legal assistant. Based on the context below, answer the user's query.
If the answer is not found in the documents, say so. Try and answer as much as you can and dont specify document number(not too concisely).

Context:
{context}

Question: {query}

Answer:"""

        try:
            print("🤖 Generating answer with Groq...")
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def query(self, question: str) -> dict:
        if not self.vectorstore:
            return {
                'question': question,
                'answer': 'Vectorstore not loaded. Please initialize first.',
                'sources': [],
                'response_time': 0
            }

        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"🔍 Query: {question}")
        print('='*60)

        docs_with_scores = self.search_similar_documents(question)
        if not docs_with_scores:
            return {
                'question': question,
                'answer': 'No relevant documents found.',
                'sources': [],
                'response_time': time.time() - start_time
            }

        answer = self.generate_answer(question, docs_with_scores)
        response_time = time.time() - start_time
        return {
            'question': question,
            'answer': answer,
            'sources': docs_with_scores,
            'response_time': response_time
        }
def main():
    rag = RAGSystem(PDF_PATH, VECTORSTORE_PATH)
    rag.create_vectorstore_from_pdf()

    while True:
        question = input("\n❓ Enter your query (or type 'exit'): ").strip()
        if question.lower() == 'exit':
            break

        result = rag.query(question)

        print("\n📝 Answer:\n", result['answer'])

        # 🔍 Print sources
        if result['sources']:
            print("\n📚 Top Sources:")
            print("-" * 40)
            for i, (doc, score) in enumerate(result['sources'], 1):
                similarity = max(0, (1 - score) * 100)
                print(f"\n🔹 Source {i} (Similarity: {similarity:.1f}%)")
                print(f"{doc.page_content[:500]}...")  # Show first 500 characters

        print(f"\n⏱️ Time: {round(result['response_time'], 2)} seconds")


if __name__ == "__main__":
    main()
