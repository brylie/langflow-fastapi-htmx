# Chat Application with FastAPI and HTMX

This project is a simple chat application built with FastAPI for the backend and HTMX for dynamic frontend updates.

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

## Setup Instructions

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/brylie/langflow-fastapi-htmx.git
cd langflow-fastapi-htmx
```

### 2. Create a Virtual Environment

#### Windows
```bash
python -m venv venv
```

#### macOS and Linux
```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment

#### Windows
```bash
venv\Scripts\activate
```

#### macOS and Linux
```bash
source venv/bin/activate
```

### 4. Install Dependencies

With the virtual environment activated, install the project dependencies:

```bash
pip install -r requirements.txt
```

### 5. Run the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn chat:app --reload
```

The `--reload` flag enables auto-reloading on code changes, which is useful for development.

### 6. Access the Application

Open your web browser and navigate to:

```
http://127.0.0.1:8000
```

You should now see the chat interface and be able to interact with the chatbot.

## Development

- The main application code is in `chat.py`.
- HTML templates are stored in the `templates` directory.
- Static files (CSS, JavaScript) are in the `static` directory.

## Troubleshooting

If you encounter any issues:

1. Ensure you're using Python 3.10 or higher.
2. Make sure your virtual environment is activated when installing dependencies and running the app.
3. Check that all required dependencies are installed correctly.
4. If you encounter any "Module not found" errors, try reinstalling the dependencies.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)
