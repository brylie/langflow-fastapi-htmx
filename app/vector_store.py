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

# Load environment variables
load_dotenv()


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

    async def query(self, query: str, top_k: int = 5) -> List[VectorStoreResult]:
        # Simulate a delay to mimic a real database query
        await asyncio.sleep(0.1)

        # Randomly select 'top_k' sentences from the lorem ipsum list
        selected_sentences = random.sample(
            self.lorem_ipsum, min(top_k, len(self.lorem_ipsum))
        )

        # Create a list of VectorStoreResult objects with the selected sentences
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


# Placeholder classes for other vector stores
class PineconeStore(VectorStore):
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

    async def query(self, query: str, top_k: int = 5) -> List[VectorStoreResult]:
        results = self.collection.query(query_texts=[query], n_results=top_k)

        vector_store_results = []

        # Process query results
        # The ChromaDB query method returns results in a specific format:
        # - 'documents', 'metadatas', and 'distances' are lists of lists
        # - The outer list corresponds to the number of queries (in our case, always 1)
        # - The inner lists contain the results for each query
        # We use [0] to access the results of our single query, then zip these lists together
        # enumerate is used to get an index (i) for each result, starting from 0
        for i, (document, metadata, distance) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            # Convert distance to a similarity score (assuming cosine distance)
            # Cosine distance ranges from 0 to 2, so we normalize and invert it
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
        """
        Filter results to ensure uniqueness based on document ID.

        :param results: List of result dictionaries from AstraDB query
        :param top_k: Maximum number of results to return
        :return: List of unique result dictionaries, up to top_k in length
        """
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

    async def query(self, query: str, top_k: int = 5) -> List[VectorStoreResult]:
        results = self.collection.find(
            sort={"$vectorize": query},
            limit=top_k,
            projection={"$vectorize": True},
            include_similarity=True,
        )

        # Filter for unique results
        unique_results = self._filter_unique_results(results, top_k)

        vector_store_results = []
        for result in unique_results:
            # Extract the document content and metadata
            # document = result.get("document", {})
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
