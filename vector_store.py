from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel, Field
import random
import asyncio


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
    async def query(self, query: str, top_k: int = 5) -> List[VectorStoreResult]:
        # Implement ChromaDB-specific query method
        pass


class AstraDBStore(VectorStore):
    async def query(self, query: str, top_k: int = 5) -> List[VectorStoreResult]:
        # Implement AstraDB-specific query method
        pass
