from typing import List
from vector_store import VectorStore
from chat_gpt_client import Message, MessageRole


class RAGService:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    async def get_relevant_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        results = await self.vector_store.query(query, top_k)
        # Process and format the results into a single context string
        context = "\n".join(
            [f"Document {i+1}: {result.content}" for i, result in enumerate(results)]
        )
        return context

    async def prepare_messages(
        self,
        system_prompt: str,
        chat_history: List[Message],
        user_message: str,
    ) -> List[Message]:
        # Get relevant context based on the user's message
        context = await self.get_relevant_context(user_message)

        # Prepare the messages in the correct order
        prepared_messages = [
            Message(role=MessageRole.system, content=system_prompt),
            *chat_history,
            Message(
                role=MessageRole.user,
                content=f"Relevant context:\n{context}\n\nUser message: {user_message}",
            ),
        ]
        return prepared_messages
