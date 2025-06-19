from typing import Dict, List
import faiss
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path

# Get absolute paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

load_dotenv()

class RAGClient:
    def __init__(self, model_path: str = None, openai_api_key: str = None):
        # 환경변수에서 API 키와 모델 경로 가져오기
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required")
            
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.model_path = model_path or str(VECTOR_STORE_DIR)
        self.vector_store = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """
        Initialize the vector store with genre-specific data
        """
        try:
            if os.path.exists(self.model_path):
                self.vector_store = FAISS.load_local(self.model_path, self.embeddings)
            else:
                # Create new vector store if it doesn't exist
                self.vector_store = FAISS.from_texts([""], self.embeddings)
        except Exception as e:
            raise Exception(f"Error initializing vector store: {str(e)}")

    def load_crawled_data(self, data_dir: str):
        """
        Load and process crawled JSON data files
        """
        all_texts = []
        
        # Load all crawl*.json files
        for filename in os.listdir(data_dir):
            if filename.startswith("crawl") and filename.endswith(".json"):
                file_path = os.path.join(data_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Extract text content from each crawled item
                        for item in data:
                            if isinstance(item, dict):
                                # Combine relevant fields into a single text
                                text_parts = []
                                if 'chunkNum' in item:
                                    text_parts.append(f"chunkNum: {item['chunkNum']}")
                                if 'content' in item:
                                    text_parts.append(f"content: {item['content']}\n")
                                    
                                combined_text = "\n".join(text_parts)
                                if combined_text.strip():
                                    all_texts.append(combined_text)

                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
                    continue
        
        return all_texts

    def create_vector_store(self, texts: List[str], chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Create a new vector store from the provided texts
        """
        try:
            # Split texts into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            print("Splitting texts into chunks...")
            all_splits = []
            for text in tqdm(texts):
                splits = text_splitter.split_text(text)
                all_splits.extend(splits)
            
            print(f"Creating vector store with {len(all_splits)} chunks...")
            # Create new vector store
            self.vector_store = FAISS.from_texts(all_splits, self.embeddings)
            
            # Save the vector store
            os.makedirs(self.model_path, exist_ok=True)
            self.vector_store.save_local(self.model_path)
            print(f"Vector store saved to {self.model_path}")
            
        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")

    def search_genre_requirements(self, genre: str) -> Dict:
        """
        Search for genre-specific requirements and guidelines
        """
        try:
            query = f"Requirements and guidelines for writing {genre} novels"
            docs = self.vector_store.similarity_search(query, k=3)
            return {
                "genre": genre,
                "requirements": [doc.page_content for doc in docs]
            }
        except Exception as e:
            raise Exception(f"Error searching genre requirements: {str(e)}")

    def search_similar_chapters(self, query: str) -> List:
        """
        Search for similar chapters based on the query
        """
        try:
            docs = self.vector_store.similarity_search(query, k=3)
            return [doc.page_content for doc in docs]
        except Exception as e:
            raise Exception(f"Error searching similar chapters: {str(e)}")

    def format_search_results(self, results: List) -> str:
        """
        Format search results into a readable string
        """
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(f"Result {i}:\n{result}\n")
        return "\n".join(formatted_results)

    def update_vector_store(self, new_texts: List[str]):
        """
        Update the vector store with new texts
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.split_text("\n".join(new_texts))
            self.vector_store.add_texts(texts)
            self.vector_store.save_local(self.model_path)
        except Exception as e:
            raise Exception(f"Error updating vector store: {str(e)}")

if __name__ == "__main__":
    try:
        print("\n=== RAG Client Test ===")
        
        # Initialize RAG client
        rag_client = RAGClient()
        print("✅ RAG Client initialized successfully")
        
        # Load crawled data
        print("\n1. Loading crawled data...")
        texts = rag_client.load_crawled_data(str(DATA_DIR))
        print(f"✅ Loaded {len(texts)} documents from crawled data")
        
        # Create vector store
        print("\n2. Creating vector store...")
        rag_client.create_vector_store(texts)
        print("✅ Vector store created successfully")
        
        # Test similarity search
        print("\n3. Testing similarity search...")
        test_query = "판타지 소설에서 마법사가 등장하는 장면"
        results = rag_client.search_similar_chapters(test_query)
        
        print("\nSearch Results:")
        print("-" * 50)
        print(rag_client.format_search_results(results))
        print("-" * 50)
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise e 