import os
import logging

from dotenv import load_dotenv
from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("CHAT_MODEL")

    if not base_url or not api_key or not model:
        print(
            "Configuration error: Check OPENAI_BASE_URL, "
            "OPENAI_API_KEY, and CHAT_MODEL in your .env file."
        )
        return

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    messages = [
        {
            "role": "system",
            "content": "You are a concise banking compliance assistant.",
        },
        {
            "role": "user",
            "content": "Explain in one sentence why regulatory compliance is important for banks.",
        },
    ]

    try:
        logging.info("REQUEST: %s", messages)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )

        answer = response.choices[0].message.content

        print("\nModel Response:")
        print(answer)

        logging.info("RESPONSE: %s", answer)

        if response.usage:
            logging.info("USAGE: %s", response.usage)

    except AuthenticationError:
        print(
            "Auth failed (401): Check OPENAI_API_KEY in your .env file."
        )

    except RateLimitError:
        print(
            "Rate limited (429): Too many requests or quota exceeded. "
            "Please retry later."
        )

    except APIConnectionError:
        print(
            "Connection failed: Could not reach the API server. "
            "Check OPENAI_BASE_URL, your internet connection, "
            "and whether the API service is running."
        )

    except Exception as error:
        print(f"Unexpected error: {error}")


if __name__ == "__main__":
    main()