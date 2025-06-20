from typing import Dict, List
import numpy as np
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

class RAGClient:
    def __init__(self, model_path: str = None, openai_api_key: str = None, force_initialize: bool = False):
        # 환경변수에서 API 키와 모델 경로 가져오기
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required")
            
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.model_path = model_path or os.getenv("VECTOR_STORE_PATH", "./vector_store")
        self.data_dir = os.getenv("DATA_DIR", "./data")
        
        if force_initialize:
            self._initialize_vector_store()
        else:
            self.vector_store = self._load_existing_vector_store()
            if self.vector_store is None:
                print("Vector store not found, initializing new vector store")
                self._initialize_vector_store()
            
    def _load_existing_vector_store(self):
        """
        Load an existing vector store from the model path
        """
        try:
            if os.path.exists(self.model_path):
                return FAISS.load_local(
                    self.model_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                raise FileNotFoundError(f"Vector store not found at {self.model_path}")
        except Exception as e:
            raise Exception(f"Error loading existing vector store: {str(e)}")

    def _initialize_vector_store(self):
        """
        Initialize the vector store with genre-specific data
        """
        try:
            print("Initializing new vector store")
            self.vector_store = FAISS.from_texts([""], self.embeddings)
            os.makedirs(self.model_path, exist_ok=True)
            self.vector_store.save_local(self.model_path)
            print(f"Vector store saved to {self.model_path}")
        except Exception as e:
            raise Exception(f"Error initializing vector store: {str(e)}")

    def load_crawled_data(self, data_dir: str = None):
        """
        Load and process crawled JSON data files
        """
        all_texts = []
        
        # Load all crawl*.json files
        data_dir = data_dir or self.data_dir
        
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
                                # if 'chunkNum' in item:
                                #     text_parts.append(f"chunkNum: {item['chunkNum']}")
                                if 'content' in item:
                                    text_parts.append(f"content: {item['content']}")
                                    
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
                chunk_overlap=0 # overlap 없음 
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

    def update_vector_store(self, new_texts: List[str]):
        """
        Update the vector store with new texts
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=0
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
        
        if rag_client.vector_store is None:
            # Load crawled data
            print("\n1. Loading crawled data...")
            texts = rag_client.load_crawled_data()
            print(f"✅ Loaded {len(texts)} documents from crawled data")
            
            # Create vector store
            print("\n2. Creating vector store...")
            rag_client.create_vector_store(texts)
            print("✅ Vector store created successfully")
            
        # Test similarity search
        print("\n3. Testing similarity search...")
        test_query = "real"
        results = rag_client.search_genre_requirements(test_query)
        
        print("\nSearch Results:")
        print("-" * 50)
        print(results['requirements'])
        print("-" * 50)
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise e 