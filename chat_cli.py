"""
chat_cli.py
-----------
An interactive Command-Line Interface (CLI) chat loop
that uses OllamaClient to talk to the local LLM, with configurable
system prompts, temperature, and multi-turn conversation history.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from ollama_client import OllamaClient

DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise AI assistant."
SAVED_CHATS_DIR = Path("saved_chats")


def save_conversation(
    history: List[Dict[str, str]],
    system_prompt: str,
    model: str,
    temperature: float,
    filename: Optional[str] = None,
) -> Path:
    """
    Persists the conversation history and session metadata to a JSON file.

    Returns:
        The Path to the saved JSON file.
    """
    SAVED_CHATS_DIR.mkdir(parents=True, exist_ok=True)

    if filename:
        safe_name = Path(filename).name
        if not safe_name.endswith(".json"):
            safe_name += ".json"
        target_path = SAVED_CHATS_DIR / safe_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_path = SAVED_CHATS_DIR / f"chat_{timestamp}.json"

    record = {
        "model": model,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "messages": history,
    }

    target_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return target_path


def resolve_system_prompt(cli_system: Optional[str]) -> str:
    """
    Determines the system prompt from CLI arguments or interactive startup input.
    Falls back to DEFAULT_SYSTEM_PROMPT if the user presses Enter.
    """
    if cli_system is not None:
        return cli_system.strip() or DEFAULT_SYSTEM_PROMPT

    user_system = input(
        f"System prompt (press Enter for default: '{DEFAULT_SYSTEM_PROMPT}'): "
    ).strip()
    return user_system if user_system else DEFAULT_SYSTEM_PROMPT


def run_chat(system_prompt: str, temperature: float, max_turns: int = 10) -> None:
    """Starts a Read-Eval-Print Loop (REPL) for chatting with Ollama."""
    client = OllamaClient(model="llama3.2:1b")
    history: List[Dict[str, str]] = []

    print(f"\n=== Local LLM Playground ({client.model}) ===")
    print(f"System prompt: {system_prompt}")
    print(f"Temperature:   {temperature}")
    print(f"Max history:   {max_turns if max_turns > 0 else 'unlimited'} turns")
    print(
        "Commands: '/system <prompt>', '/temp <value>', '/history', '/clear', "
        "'/save [name]', 'exit' or 'quit' to stop.\n"
    )

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Exiting chat. Goodbye!")
            break

        if user_input.startswith("/system"):
            new_prompt = user_input[len("/system"):].strip()
            if new_prompt:
                system_prompt = new_prompt
                print(f"[System prompt updated]: {system_prompt}\n")
            else:
                print(f"[Current system prompt]: {system_prompt}\n")
            continue

        if user_input.startswith("/temp"):
            raw_value = user_input[len("/temp"):].strip()
            if not raw_value:
                print(f"[Current temperature]: {temperature}\n")
                continue
            try:
                new_temp = float(raw_value)
                if new_temp < 0:
                    print("[Error] Temperature must be >= 0.0\n")
                else:
                    temperature = new_temp
                    print(f"[Temperature updated]: {temperature}\n")
            except ValueError:
                print("[Error] Please provide a valid number, e.g. '/temp 0.2'\n")
            continue

        if user_input == "/clear":
            history.clear()
            print("[Conversation history cleared]\n")
            continue

        if user_input == "/history":
            print(f"[History: {len(history) // 2} turns ({len(history)} messages)]")
            for msg in history:
                print(f"  {msg['role']}: {msg['content']}")
            print()
            continue

        if user_input.startswith("/save"):
            if not history:
                print("[Nothing to save yet — send a message first]\n")
                continue
            custom_name = user_input[len("/save"):].strip() or None
            saved_path = save_conversation(
                history=history,
                system_prompt=system_prompt,
                model=client.model,
                temperature=temperature,
                filename=custom_name,
            )
            print(f"[Conversation saved to {saved_path}]\n")
            continue

        history.append({"role": "user", "content": user_input})

        # Keep system prompt separate at index 0 so it is never trimmed by sliding window
        messages = [{"role": "system", "content": system_prompt}] + history

        try:
            response = client.chat(messages=messages, temperature=temperature)
            reply = response["message"]["content"].strip()
            history.append({"role": "assistant", "content": reply})

            if max_turns > 0 and len(history) > max_turns * 2:
                history = history[-(max_turns * 2):]

            prompt_tokens = response.get("prompt_eval_count", 0)
            print(f"\nAssistant: {reply}")
            print(f"[tokens in context: {prompt_tokens}]\n")
        except Exception as err:
            # Remove the unanswered user message so history stays alternating user/assistant
            history.pop()
            print(f"\n[Error] {err}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local LLM Playground CLI")
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="Optional system prompt to guide the assistant's behavior.",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (>= 0.0). Lower = deterministic, higher = creative.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Number of recent conversation turns to keep in memory (0 = unlimited).",
    )
    args = parser.parse_args()

    if args.temperature < 0:
        parser.error("--temperature must be >= 0.0")
    if args.max_turns < 0:
        parser.error("--max-turns must be >= 0")

    chosen_system_prompt = resolve_system_prompt(args.system)
    run_chat(
        system_prompt=chosen_system_prompt,
        temperature=args.temperature,
        max_turns=args.max_turns,
    )
