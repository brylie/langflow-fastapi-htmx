import httpx
from typing import Optional, Dict, Any

BASE_API_URL = "http://127.0.0.1:7860/api/v1/run"
FLOW_ID = "8c4e757e-4bbf-45b0-b131-14a3e9af1836"
ENDPOINT = ""  # You can set a specific endpoint name in the flow settings

# You can tweak the flow by adding a tweaks dictionary
TWEAKS = {
    "Memory-FvisK": {},
    "Prompt-e7qkR": {},
    "ChatInput-ZVCpy": {},
    "OpenAIModel-YWZWf": {},
    "ChatOutput-gADpg": {},
}


async def run_flow(
    message: str,
    endpoint: str = ENDPOINT or FLOW_ID,
    output_type: str = "chat",
    input_type: str = "chat",
    tweaks: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a flow with a given message and optional tweaks asynchronously.
    :param message: The message to send to the flow
    :param endpoint: The ID or the endpoint name of the flow
    :param output_type: The type of output expected (default is "chat")
    :param input_type: The type of input being sent (default is "chat")
    :param tweaks: Optional tweaks to customize the flow
    :param api_key: Optional API key for authentication
    :return: The JSON response from the flow
    """
    api_url = f"{BASE_API_URL}/{endpoint}"
    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }
    headers = {}
    if tweaks:
        payload["tweaks"] = tweaks
    if api_key:
        headers["x-api-key"] = api_key

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(api_url, json=payload, headers=headers)
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e}")
            return {"error": str(e)}
        except httpx.RequestError as e:
            print(f"An error occurred while requesting: {e}")
            return {"error": str(e)}


async def get_chat_response(message: str) -> str:
    """
    Asynchronous function to get a chat response from the LLM.
    :param message: The user's message
    :return: The LLM's response as a string
    """
    response = await run_flow(message, tweaks=TWEAKS)

    if "error" in response:
        return f"I'm sorry, but I encountered an error: {response['error']}"

    try:
        # Navigate through the nested structure to get the message text
        llm_response = response["outputs"][0]["outputs"][0]["results"]["message"][
            "text"
        ]
    except (KeyError, IndexError) as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Full response: {response}")
        return "I'm sorry, but I couldn't understand the response. Please try again."

    return llm_response
