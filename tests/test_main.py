import pytest
from fastapi.testclient import TestClient
from app.main import app
from bs4 import BeautifulSoup

client = TestClient(app)

# Mock data
mock_rag_citations = [
    {"source": "Source 1", "content": "Content 1"},
    {"source": "Source 2", "content": "Content 2"},
]

mock_chat_response = "This is a mock response from the LLM."


# Mock functions
async def mock_prepare_messages_with_sources(*args, **kwargs):
    return [], mock_rag_citations


async def mock_get_chat_response_with_history(*args, **kwargs):
    return mock_chat_response


@pytest.fixture
def load_env_variables(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")


@pytest.fixture
def mock_services(monkeypatch, load_env_variables):
    # Mock RAG service
    monkeypatch.setattr(
        "app.main.rag_service.prepare_messages_with_sources",
        mock_prepare_messages_with_sources,
    )

    # Mock LLM service (get_chat_response_with_history)
    monkeypatch.setattr(
        "app.main.get_chat_response_with_history", mock_get_chat_response_with_history
    )


def test_sources_toggle_independent(mock_services):
    messages = [
        "Tell me about prompt engineering",
        "What are some key aspects of prompt engineering?",
        "How is prompt engineering used in AI?",
    ]

    all_ids = []
    for i, message in enumerate(messages):
        response = client.post("/chat", data={"message": message})
        assert (
            response.status_code == 200
        ), f"Failed to get response for message: {message}"

        soup = BeautifulSoup(response.content, "html.parser")
        toggles = soup.find_all("button", attrs={"data-bs-toggle": "collapse"})

        print(f"Number of toggles found for message '{message}': {len(toggles)}")

        for toggle in toggles:
            print(f"Toggle attributes:")
            for attr, value in toggle.attrs.items():
                print(f"  {attr}: {value}")

            target = toggle.get("data-bs-target")
            if target:
                id_match = target.split("-")[-1]  # Assuming format '#sources-<id>'
                all_ids.append(id_match)
                print(f"ID found: {id_match}")
            else:
                print("Toggle found without data-bs-target attribute")

        # Print the entire HTML content for debugging
        print(f"Full HTML content for message {i + 1}:")
        print(soup.prettify())
        print("\n" + "=" * 50 + "\n")

    unique_ids = set(all_ids)
    print(f"All IDs found: {all_ids}")
    print(f"Unique IDs found: {unique_ids}")

    assert (
        len(unique_ids) > 1
    ), f"Expected multiple unique IDs, but found {len(unique_ids)}: {unique_ids}"
    assert len(unique_ids) == len(
        all_ids
    ), f"Some IDs are not unique. All IDs: {all_ids}, Unique IDs: {unique_ids}"
