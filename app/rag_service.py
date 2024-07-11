from typing import List, Tuple
import agentops

from app.vector_store import VectorStore
from app.chat_gpt_client import Message, MessageRole
from app.models import RagCitation

class RAGService:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    @agentops.record_function('get_relevant_context')
    async def get_relevant_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> Tuple[str, List[RagCitation]]:
        try:
            results = await self.vector_store.query(query, top_k)
            context = "\n".join([result.content for result in results])
            citations = [
                RagCitation(source=result.metadata.source, content=result.content)
                for result in results
            ]
            return context, citations
        except Exception as e:
            agentops.log_error(f"Error in get_relevant_context: {str(e)}")
            raise

    @agentops.record_function('prepare_messages_with_sources')
    async def prepare_messages_with_sources(
        self,
        system_prompt: str,
        chat_history: List[Message],
        user_message: str,
    ) -> Tuple[List[Message], List[RagCitation]]:
        try:
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
        except Exception as e:
            agentops.log_error(f"Error in prepare_messages_with_sources: {str(e)}")
            raise

    @agentops.record_function('prepare_messages')
    async def prepare_messages(
        self, system_prompt: str, chat_history: List[Message], user_message: str
    ) -> List[Message]:
        try:
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
        except Exception as e:
            agentops.log_error(f"Error in prepare_messages: {str(e)}")
            raise

# Example usage and testing
@agentops.record_function('main')
async def main():
    # This is a placeholder for testing purposes
    # You would typically initialize your vector store and RAGService here
    from app.vector_store import MockVectorStore
    vector_store = MockVectorStore()
    rag_service = RAGService(vector_store)

    # Example usage
    system_prompt = "You are a helpful assistant."
    chat_history = [
        Message(role=MessageRole.user, content="Hello!"),
        Message(role=MessageRole.assistant, content="Hi there! How can I help you today?"),
    ]
    user_message = "Tell me about the weather."

    try:
        prepared_messages, citations = await rag_service.prepare_messages_with_sources(
            system_prompt, chat_history, user_message
        )
        print("Prepared Messages:", prepared_messages)
        print("Citations:", citations)

        prepared_messages_without_sources = await rag_service.prepare_messages(
            system_prompt, chat_history, user_message
        )
        print("Prepared Messages (without sources):", prepared_messages_without_sources)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(main())
    except Exception as e:
        agentops.log_error(str(e))
    finally:
        agentops.end_session('Success')