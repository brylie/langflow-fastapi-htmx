import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_sources_toggle_independent():
    """
    Test to verify that each message's sources toggle independently.
    """
    # Send a chat request to trigger a response containing the sources toggle
    response = client.post("/chat", data={"message": "Hello"})
    assert response.status_code == 200

    # Extract the response HTML
    response_html = response.text

    # Check if the sources toggle contains the correct Bootstrap attributes
    assert 'data-bs-toggle="collapse"' in response_html
    assert 'data-bs-target="#sources-' in response_html
    assert 'aria-expanded="false"' in response_html
    assert 'aria-controls="sources-' in response_html

    # Verify that each sources container has a unique ID
    unique_ids = set()
    start = 0
    while True:
        start = response_html.find('data-bs-target="#sources-', start)
        if start == -1:
            break
        end = response_html.find('"', start + 23)
        unique_id = response_html[start + 23:end]
        unique_ids.add(unique_id)
        start = end + 1

    assert len(unique_ids) > 1  # Assuming there's more than one message in the response
