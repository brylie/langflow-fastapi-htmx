import asyncio
import os
from abc import ABC, abstractmethod
import random
from typing import Any, Dict, List
from astrapy import DataAPIClient
from pydantic import BaseModel, Field
import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
import agentops

# Load environment variables
load_dotenv()

# Initialize AgentOps
AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")
if not AGENTOPS_API_KEY:
    raise ValueError("AGENTOPS_API_KEY not found in environment variables")

agentops.init(AGENTOPS_API_KEY)

class VectorStoreMetadata(BaseModel):
    score: float = Field(..., description="Relevance score of the document")
    source: str = Field(..., description="Source of the document")

class VectorStoreResult(BaseModel):
    content: str = Field(..., description="Content of the document")
    metadata: VectorStoreMetadata

class VectorStore(ABC):
    @abstractmethod
    async def query(self, query: str, top_k: int = 5) -> List[VectorStoreResult]:
        """
        Query the vector store and return the top k most relevant documents.

        :param query: The query string
        :param top_k: Number of top results to return
        :return: List of VectorStoreResult objects containing document content and metadata
        """
        pass

class MockVectorStore(VectorStore):
    def __init__(self):
        self.lorem_ipsum = [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
            "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum.",
            "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia.",
            "Nisi ut aliquip ex ea commodo consequat.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod.",
            "Tempor incididunt ut labore et dolore magna aliqua.",
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco.",
            "Laboris nisi ut aliquip ex ea commodo consequat.",
        ]

    @agentops.record_function('MockVectorStore.query')
    async def query(self, query: str, top_k: int = 5) -> List[VectorStoreResult]:
        try:
            await asyncio.sleep(0.1)
            selected_sentences = random.sample(
                self.lorem_ipsum, min(top_k, len(self.lorem_ipsum))
            )
            results = [
                VectorStoreResult(
                    content=sentence,
                    metadata=VectorStoreMetadata(
                        score=round(random.uniform(0.5, 1.0), 2),
                        source=f"mock_document_{index_n+1}.txt",
                    ),
                )
                for index_n, sentence in enumerate(selected_sentences)
            ]
            return results
        except Exception as e:
            agentops.log_error(f"Error in MockVectorStore query: {str(e)}")
            raise

class PineconeStore(VectorStore):
    @agentops.record_function('PineconeStore.query')
    async def query(self, query: str, top_k: int = 5) -> List[VectorStoreResult]:
        # Implement Pinecone-specific query method
        pass

class ChromaDBStore(VectorStore):
    def __init__(self, path: str, collection_name: str = "default_collection"):
        self.client = chromadb.PersistentClient(
            path=path, settings=Settings(allow_reset=True)
        )
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=openai_ef
        )

    @agentops.record_function('ChromaDBStore.query')
    async def query(self, query: str, top_k: int = 5) -> List[VectorStoreResult]:
        try:
            results = self.collection.query(query_texts=[query], n_results=top_k)
            vector_store_results = []
            for i, (document, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                similarity_score = 1 - (distance / 2)
                vector_store_results.append(
                    VectorStoreResult(
                        content=document,
                        metadata=VectorStoreMetadata(
                            score=similarity_score,
                            source=metadata.get("source", f"document_{i}"),
                        ),
                    )
                )
            return vector_store_results
        except Exception as e:
            agentops.log_error(f"Error in ChromaDBStore query: {str(e)}")
            raise

class AstraDBStore(VectorStore):
    def __init__(self, collection_name: str = "default_collection"):
        self.astra_db_endpoint = os.getenv("ASTRA_DB_ENDPOINT")
        self.astra_db_token = os.getenv("ASTRA_DB_TOKEN")
        if not self.astra_db_endpoint or not self.astra_db_token:
            raise ValueError(
                "ASTRA_DB_ENDPOINT and ASTRA_DB_TOKEN must be set in the environment"
            )
        self.client = DataAPIClient(token=self.astra_db_token)
        self.db = self.client.get_database_by_api_endpoint(self.astra_db_endpoint)
        self.collection = self.db.get_collection(collection_name)

    def _filter_unique_results(
        self,
        results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        seen_content = set()
        unique_results = []
        for result in results:
            content = result.get("content")
            if content not in seen_content:
                seen_content.add(content)
                unique_results.append(result)
                if len(unique_results) == top_k:
                    break
        return unique_results

    @agentops.record_function('AstraDBStore.query')
    async def query(self, query: str, top_k: int = 5) -> List[VectorStoreResult]:
        try:
            results = self.collection.find(
                sort={"$vectorize": query},
                limit=top_k,
                projection={"$vectorize": True},
                include_similarity=True,
            )
            unique_results = self._filter_unique_results(results, top_k)
            vector_store_results = []
            for result in unique_results:
                content = result.get("content", "")
                similarity = result.get("$similarity", 0.0)
                metadata = result.get("metadata", {})
                source = metadata.get("source", "Unknown")
                vector_store_results.append(
                    VectorStoreResult(
                        content=content,
                        metadata=VectorStoreMetadata(score=similarity, source=source),
                    )
                )
            return vector_store_results
        except Exception as e:
            agentops.log_error(f"Error in AstraDBStore query: {str(e)}")
            raise

@agentops.record_function('main')
async def main():
    # This is a placeholder for testing purposes
    mock_store = MockVectorStore()
    chroma_store = ChromaDBStore(path="./chromadb")
    astra_store = AstraDBStore()

    query = "example query"
    top_k = 3

    try:
        print("MockVectorStore results:")
        mock_results = await mock_store.query(query, top_k)
        for result in mock_results:
            print(f"Content: {result.content}")
            print(f"Score: {result.metadata.score}")
            print(f"Source: {result.metadata.source}")
            print()

        print("ChromaDBStore results:")
        chroma_results = await chroma_store.query(query, top_k)
        for result in chroma_results:
            print(f"Content: {result.content}")
            print(f"Score: {result.metadata.score}")
            print(f"Source: {result.metadata.source}")
            print()

        print("AstraDBStore results:")
        astra_results = await astra_store.query(query, top_k)
        for result in astra_results:
            print(f"Content: {result.content}")
            print(f"Score: {result.metadata.score}")
            print(f"Source: {result.metadata.source}")
            print()

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        agentops.log_error(str(e))
    finally:
        agentops.end_session('Success')