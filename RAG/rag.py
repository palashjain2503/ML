# RAG System with Persistent ChromaDB Storage
# Requirements: pip install langchain chromadb sentence-transformers google-generativeai pypdf
# groq-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
import os
import time
import shutil
from typing import List, Optional
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Configuration
GOOGLE_API_KEY = "AIzaSyCBTBocgeKCRxLg_2Cy9SpnU9X2lrg9vQw"  # Replace with your key
# filepath: c:\Users\palas\Desktop\programs\project\utilities\RAG\rag.py
# ...existing code...
PDF_PATH = r"C:\Users\palas\Desktop\programs\project\utilities\RAG\IPC.pdf"  
# Use raw string
VECTORSTORE_PATH = r"C:\Users\palas\Desktop\programs\project\utilities\RAG\rag_vectorstore1"  # Persistent storage path
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4

# Set up Gemini
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

class RAGSystem:
    def __init__(self, pdf_path: str, vectorstore_path: str):
        self.pdf_path = pdf_path
        self.vectorstore_path = vectorstore_path
        self.vectorstore = None
        self.embeddings = None
        self.llm = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=1500,
                temperature=0.2,
                top_p=0.8
            )
        )
        
    def setup_embeddings(self):
        """Initialize embedding model"""
        if self.embeddings is None:
            print("🔍 Setting up embeddings...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-base-en-v1.5",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✅ Embeddings ready!")
    
    def vectorstore_exists(self) -> bool:
        """Check if vectorstore already exists"""
        return os.path.exists(self.vectorstore_path) and os.listdir(self.vectorstore_path)
    
    def create_vectorstore_from_pdf(self, force_rebuild: bool = False) -> bool:
        """Create vectorstore from PDF (only if doesn't exist or force rebuild)"""
        
        if self.vectorstore_exists() and not force_rebuild:
            print(f"📂 Vectorstore already exists at {self.vectorstore_path}")
            choice = input("Do you want to rebuild it? (y/N): ").lower()
            if choice != 'y':
                print("✅ Using existing vectorstore")
                return True
        
        if force_rebuild and self.vectorstore_exists():
            print("🗑️ Removing old vectorstore...")
            shutil.rmtree(self.vectorstore_path)
            time.sleep(1)
        
        print("🏗️ Building vectorstore from PDF...")
        
        # Load and chunk PDF
        chunks = self._load_and_chunk_pdf()
        if not chunks:
            return False
        
        # Setup embeddings
        self.setup_embeddings()
        
        # Create vectorstore
        return self._create_vectorstore(chunks)
    
    def _load_and_chunk_pdf(self) -> List[Document]:
        """Load PDF and split into chunks"""
        print(f"📖 Loading PDF: {self.pdf_path}")
        
        try:
            loader = PyPDFLoader(self.pdf_path)
            pages = loader.load()
            print(f"✅ Loaded {len(pages)} pages")
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            
            chunks = splitter.split_documents(pages)
            print(f"✅ Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
            
            return chunks
            
        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
            return []
    
    def _create_vectorstore(self, chunks: List[Document]) -> bool:
        """Create ChromaDB vectorstore with persistence"""
        print(f"💾 Creating persistent vectorstore at {self.vectorstore_path}...")
        
        try:
            # Create vectorstore with persistence
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.vectorstore_path
            )
            
            print(f"✅ Vectorstore created with {len(chunks)} documents")
            print(f"📁 Saved to: {self.vectorstore_path}")
            return True
            
        except Exception as e:
            print(f"❌ ChromaDB creation failed: {e}")
            print("🔄 Trying batch approach...")
            return self._create_vectorstore_batch(chunks)
    
    def _create_vectorstore_batch(self, chunks: List[Document]) -> bool:
        """Fallback: Create vectorstore in batches"""
        try:
            # Create empty vectorstore
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.vectorstore_path
            )
            
            # Add documents in batches
            batch_size = 25
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(chunks) + batch_size - 1) // batch_size
                
                print(f"📥 Adding batch {batch_num}/{total_batches} ({len(batch)} docs)")
                
                try:
                    self.vectorstore.add_documents(batch)
                    time.sleep(0.1)
                except Exception as batch_error:
                    print(f"⚠️ Batch {batch_num} failed: {batch_error}")
                    continue
            
            # Persist the vectorstore
            self.vectorstore.persist()
            print("✅ Batch creation completed and persisted!")
            return True
            
        except Exception as e:
            print(f"❌ Batch creation failed: {e}")
            return False
    
    def load_existing_vectorstore(self) -> bool:
        """Load existing vectorstore from disk"""
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
            
            # Test the vectorstore
            collection = self.vectorstore._collection
            doc_count = collection.count()
            print(f"✅ Loaded vectorstore with {doc_count} documents")
            return True
            
        except Exception as e:
            print(f"❌ Error loading vectorstore: {e}")
            return False
    
    def search_similar_documents(self, query: str) -> List[tuple]:
        """Search and return top similar documents with scores"""
        if not self.vectorstore:
            return []
        
        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query, 
                k=TOP_K
            )
            
            # Sort by similarity score (lower = more similar)
            sorted_docs = sorted(docs_with_scores, key=lambda x: x[1])
            
            print(f"🔍 Found {len(sorted_docs)} similar documents (sorted by relevance)")
            return sorted_docs
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def generate_answer(self, query: str, docs_with_scores: List[tuple]) -> str:
        """Generate answer using Gemini"""
        if not docs_with_scores:
            return "No relevant documents found."
        
        context_parts = []
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            similarity_percent = max(0, (1 - score) * 100)
            context_parts.append(
                f"Document {i} (Similarity: {similarity_percent:.1f}%):\n{doc.page_content[:600]}"
            )
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are a helpful assistant. Answer the question based on the provided documents.
If the answer is not in the documents, say so clearly.

Context from {len(docs_with_scores)} most relevant documents:
{context}

Question: {query}

Answer (be specific and cite relevant information):"""

        try:
            print("🤖 Generating answer with Gemini...")
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def query(self, question: str) -> dict:
        """Main query function"""
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

def show_menu():
    """Show the main menu"""
    print("\n🤖 RAG System Menu")
    print("=" * 40)
    print("1. Initialize/Build vectorstore from PDF")
    print("2. Load existing vectorstore")
    print("3. Query the system")
    print("4. Rebuild vectorstore (force)")
    print("5. Show vectorstore info")
    print("6. Exit")
    print("=" * 40)

def main():
    """Main function with menu system"""
    rag = RAGSystem(PDF_PATH, VECTORSTORE_PATH)
    
    print("🚀 RAG System with Persistent Storage")
    print(f"📋 Config: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, top_k={TOP_K}")
    print(f"📁 Vectorstore path: {VECTORSTORE_PATH}")
    
    while True:
        try:
            show_menu()
            choice = input("\nChoose an option (1-6): ").strip()
            
            if choice == '1':
                print("\n🏗️ Initializing vectorstore...")
                if rag.create_vectorstore_from_pdf():
                    print("✅ Vectorstore ready!")
                else:
                    print("❌ Failed to create vectorstore")
            
            elif choice == '2':
                print("\n📂 Loading existing vectorstore...")
                if rag.load_existing_vectorstore():
                    print("✅ Vectorstore loaded!")
                else:
                    print("❌ Failed to load vectorstore")
            
            elif choice == '3':
                if not rag.vectorstore:
                    print("⚠️ Please initialize or load vectorstore first (options 1 or 2)")
                    continue
                
                print("\n🤖 Query Mode - Type 'back' to return to menu")
                while True:
                    question = input("\n❓ Your question: ").strip()
                    
                    if question.lower() == 'back':
                        break
                    
                    if not question:
                        continue
                    
                    result = rag.query(question)
                    
                    print(f"\n📝 Answer:")
                    print("-" * 40)
                    print(result['answer'])
                    
                    if result['sources']:
                        print(f"\n📚 Sources (Top {len(result['sources'])}):")
                        print("-" * 40)
                        for i, (doc, score) in enumerate(result['sources'], 1):
                            similarity = max(0, (1 - score) * 100)
                            print(f"\n{i}. Similarity: {similarity:.1f}%")
                            print(f"Content: {doc.page_content[:300]}...")
                    
                    print(f"\n⏱️ Response time: {result['response_time']:.2f}s")
            
            elif choice == '4':
                print("\n🔄 Rebuilding vectorstore...")
                if rag.create_vectorstore_from_pdf(force_rebuild=True):
                    print("✅ Vectorstore rebuilt!")
                else:
                    print("❌ Failed to rebuild vectorstore")
            
            elif choice == '5':
                if rag.vectorstore:
                    try:
                        count = rag.vectorstore._collection.count()
                        print(f"\n📊 Vectorstore Info:")
                        print(f"Documents: {count}")
                        print(f"Path: {VECTORSTORE_PATH}")
                        print(f"Exists on disk: {rag.vectorstore_exists()}")
                    except:
                        print("⚠️ Could not get vectorstore info")
                else:
                    print("⚠️ No vectorstore loaded")
            
            elif choice == '6':
                print("👋 Goodbye!")
                break
            
            else:
                print("⚠️ Invalid choice. Please select 1-6.")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()