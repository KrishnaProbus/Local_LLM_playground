"""
ollama_client.py
----------------
A lightweight, dependency-minimal Python client for communicating with
a local Ollama instance via its native REST API.
"""

from typing import Any, Dict, List
import requests


class OllamaClient:
    """
    Communicates with Ollama's local HTTP server.
    Default endpoint: http://localhost:11434
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:1b"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.chat_endpoint = f"{self.base_url}/api/chat"

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Sends a conversation payload to Ollama's /api/chat endpoint.

        Parameters:
            messages:    List of message dictionaries, each with 'role' and 'content'.
                         e.g. [{"role": "user", "content": "Hello!"}]
            temperature: Controls randomness (0.0 = deterministic, >1.0 = more creative).
            stream:      Whether to stream response chunk-by-chunk. Set to False for now.

        Returns:
            The parsed JSON response dictionary from Ollama containing 'message' and stats.
        """
        if temperature < 0:
            raise ValueError("temperature must be >= 0.0")

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
            },
        }

        try:
            response = requests.post(self.chat_endpoint, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Is the Ollama service running? Start it from your Start Menu or system tray."
            )
        except requests.exceptions.HTTPError as err:
            raise RuntimeError(f"Ollama API returned an HTTP error: {err}")


if __name__ == "__main__":
    print("Testing OllamaClient with model 'llama3.2:1b'...\n")
    client = OllamaClient(model="llama3.2:1b")

    test_messages = [
        {"role": "user", "content": "In one short sentence, what is an API?"}
    ]

    result = client.chat(messages=test_messages)
    assistant_reply = result["message"]["content"]
    print("Assistant Response:")
    print(assistant_reply.strip())

    # Engineering metrics from Ollama's response
    eval_count = result.get("eval_count", 0)
    eval_duration_sec = result.get("eval_duration", 1) / 1e9
    tokens_per_sec = eval_count / eval_duration_sec if eval_duration_sec > 0 else 0

    print("\n--- Engineering Metrics ---")
    print(f"Generated tokens: {eval_count}")
    print(f"Generation time:  {eval_duration_sec:.2f} seconds")
    print(f"Speed:            {tokens_per_sec:.1f} tokens/second")

    