from typing import List, Tuple

from app.vector_store import VectorStore
from app.chat_gpt_client import Message, MessageRole
from app.models import RagCitation


class RAGService:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    async def get_relevant_context(
        self, query: str, top_k: int = 5
    ) -> Tuple[str, List[RagCitation]]:
        results = await self.vector_store.query(query, top_k)
        context = "\n".join([result.content for result in results])
        citations = [
            RagCitation(source=result.metadata.source, content=result.content)
            for result in results
        ]
        return context, citations

    async def prepare_messages_with_sources(
        self, system_prompt: str, chat_history: List[Message], user_message: str
    ) -> Tuple[List[Message], List[RagCitation]]:
        context, citations = await self.get_relevant_context(user_message)

        prepared_messages = [
            Message(
                role=MessageRole.system,
                content=f"{system_prompt}\n\nRelevant context: {context}",
            ),
            *chat_history,
            Message(role=MessageRole.user, content=user_message),
        ]

        return prepared_messages, citations

    # Keep the original prepare_messages method for backwards compatibility
    async def prepare_messages(
        self, system_prompt: str, chat_history: List[Message], user_message: str
    ) -> List[Message]:
        context, _ = await self.get_relevant_context(user_message)

        prepared_messages = [
            Message(
                role=MessageRole.system,
                content=f"{system_prompt}\n\nRelevant context: {context}",
            ),
            *chat_history,
            Message(role=MessageRole.user, content=user_message),
        ]

        return prepared_messages
