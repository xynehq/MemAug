"""
Utility script to test LLM connection and configuration.
"""
import os
import sys
import asyncio

try:
    import openai
    from dotenv import load_dotenv
    load_dotenv()
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


async def test_llm_connection():
    """
    Test the LLM connection with a simple inference.
    """
    if not HAS_OPENAI:
        print("\n❌ Error: OpenAI library not installed.")
        print("\nInstall with:")
        print("  uv pip install openai python-dotenv")
        print("\nOr install optional dependencies:")
        print("  uv pip install -e \".[tasks]\"")
        return False

    # Load configuration
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model_name = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")

    print("\n" + "="*60)
    print("LLM Configuration Test")
    print("="*60)
    print(f"\n📋 Configuration:")
    print(f"  API Key: {'✓ Set' if api_key else '✗ Not set'}")
    print(f"  Base URL: {base_url if base_url else 'Default (OpenAI)'}")
    print(f"  Model: {model_name}")

    if not api_key:
        print("\n❌ Error: OPENAI_API_KEY not set in environment")
        print("\nCreate a .env file with:")
        print("  OPENAI_API_KEY=your_key_here")
        print("  LLM_BASE_URL=https://api.openai.com/v1  (optional)")
        print("  LLM_MODEL_NAME=gpt-3.5-turbo  (optional)")
        return False

    print(f"\n🔄 Testing connection...")

    try:
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        # Simple test query
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, the LLM is working!' in JSON format with a 'message' field."}
            ],
            temperature=0.1,
            max_tokens=50
        )

        result = response.choices[0].message.content

        print(f"\n✅ Connection successful!")
        print(f"\n📤 Response from {model_name}:")
        print(f"  {result}")

        # Show usage stats if available
        if hasattr(response, 'usage'):
            print(f"\n📊 Token usage:")
            print(f"  Prompt tokens: {response.usage.prompt_tokens}")
            print(f"  Completion tokens: {response.usage.completion_tokens}")
            print(f"  Total tokens: {response.usage.total_tokens}")

        print("\n" + "="*60)
        print("✓ LLM is ready to use!")
        print("="*60 + "\n")

        return True

    except openai.AuthenticationError:
        print("\n❌ Authentication Error: Invalid API key")
        print("  Check your OPENAI_API_KEY in .env file")
        return False

    except openai.APIConnectionError as e:
        print(f"\n❌ Connection Error: {e}")
        print("  Check your LLM_BASE_URL in .env file")
        print("  Make sure the API endpoint is accessible")
        return False

    except openai.RateLimitError:
        print("\n❌ Rate Limit Error: Too many requests")
        print("  Wait a moment and try again")
        return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"  Type: {type(e).__name__}")
        return False


def cli():
    """Command-line interface for testing LLM."""
    result = asyncio.run(test_llm_connection())
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    cli()
