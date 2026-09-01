import os
from dotenv import load_dotenv

load_dotenv()

print("RegulSense RAG Environment Setup Successful!")
print("Environment variables loaded successfully.")

print(f"Chat Model: {os.getenv('CHAT_MODEL')}")
print(f"Embedding Model: {os.getenv('EMBEDDING_MODEL')}")