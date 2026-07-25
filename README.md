# George Fox Writings Chat Application

This project is a RAG (Retrieval-Augmented Generation) based chat application focused on George Fox and Quakerism. It uses FastAPI for the backend and HTMX for dynamic frontend updates, creating a responsive chat interface that provides information about George Fox's writings and Quaker history.

## Features

- Interactive chat interface with a bot that answers questions about George Fox and Quakerism
- RAG (Retrieval-Augmented Generation) architecture that retrieves information from a knowledge base
- Citation management showing the sources of information provided
- Smooth animations for messages with typing indicators
- Responsive design with Bootstrap styling

## Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) (Python package and project manager)
- OpenAI API key (for GPT-4o or other specified model)

## Setup Instructions

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/WesternFriend/george-fox-rag-chat.git
cd george-fox-rag-chat
```

### 2. Install Dependencies

`uv` creates and manages the virtual environment automatically:

```bash
uv sync
```

### 3. Configure Environment Variables

```bash
cp .env.example .env   # then set OPENAI_API_KEY
```

### 4. Run the Application

Start the FastAPI server using Uvicorn:

```bash
uv run uvicorn app.main:app --reload
```

The `--reload` flag enables auto-reloading on code changes, which is useful for development.

### 5. Access the Application

Open your web browser and navigate to:

```
http://127.0.0.1:8000
```

You should now see the chat interface and be able to interact with the chatbot.

## Continuous Integration (CI) Process

We use GitHub Actions to automatically run our pytest suite on every pull request to the main branch. This ensures that all tests pass before changes can be merged.

### CI Status Badge

![CI Status](https://github.com/brylie/langflow-fastapi-htmx/actions/workflows/pytest.yml/badge.svg)

## Development

- The main application code is in `chat.py`.
- HTML templates are stored in the `templates` directory.
- Static files (CSS, JavaScript) are in the `static` directory.

## Running Tests

To run tests, use the following command:

```bash
uv run pytest
```

This will execute all tests in the `tests/` directory. For measuring test coverage, you can use `pytest-cov` by running:

```bash
uv run pytest --cov=app
```

This command will provide a report on the test coverage for the application.

## Troubleshooting

If you encounter any issues:

1. Ensure you're using Python 3.10 or higher.
2. Check that all required dependencies are installed correctly by re-running `uv sync`.
3. If you encounter any "Module not found" errors, try re-running `uv sync`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)
