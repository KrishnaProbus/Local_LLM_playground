"""
compare_models.py
-----------------
Sends the exact same prompt, system prompt, and temperature to two
local Ollama models and compares their responses and performance metrics.
"""

import argparse
import time
from typing import List
from ollama_client import OllamaClient

DEFAULT_MODELS = ["llama3.2:1b", "qwen2.5:0.5b"]
DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise AI assistant."


def compare_models(
    models: List[str],
    prompt: str,
    system_prompt: str,
    temperature: float,
) -> None:
    """Runs the same prompt against each model and prints responses + speed metrics."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    print("\n=== Model Comparison ===")
    print(f"System prompt: {system_prompt}")
    print(f"Temperature:   {temperature}")
    print(f"User prompt:   {prompt}\n")

    for model_name in models:
        client = OllamaClient(model=model_name)
        print(f"--- Model: {model_name} ---")

        try:
            start = time.perf_counter()
            result = client.chat(messages=messages, temperature=temperature)
            elapsed_sec = time.perf_counter() - start
        except Exception as err:
            print(f"[Error] {err}")
            print(f"Tip: Make sure '{model_name}' is downloaded via: ollama pull {model_name}\n")
            continue

        reply = result["message"]["content"].strip()
        eval_count = result.get("eval_count", 0)
        eval_duration_sec = result.get("eval_duration", 0) / 1e9
        tokens_per_sec = eval_count / eval_duration_sec if eval_duration_sec > 0 else 0.0

        print(reply)
        print(
            f"\n[Metrics] tokens={eval_count} | "
            f"speed={tokens_per_sec:.1f} tok/s | "
            f"total_wall_time={elapsed_sec:.2f}s\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare responses from two local Ollama models.")
    parser.add_argument(
        "--models",
        nargs=2,
        default=DEFAULT_MODELS,
        metavar=("MODEL_A", "MODEL_B"),
        help="Two Ollama model names to compare (default: llama3.2:1b qwen2.5:0.5b).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt to send to both models. If omitted, you will be asked interactively.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used for both models.",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (>= 0.0) used for both models.",
    )
    args = parser.parse_args()

    if args.temperature < 0:
        parser.error("--temperature must be >= 0.0")

    user_prompt = args.prompt.strip() if args.prompt else input("Enter prompt to compare: ").strip()
    if not user_prompt:
        parser.error("Prompt cannot be empty.")

    compare_models(
        models=args.models,
        prompt=user_prompt,
        system_prompt=args.system,
        temperature=args.temperature,
    )
