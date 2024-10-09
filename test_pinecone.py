import os
import pinecone
from dotenv import load_dotenv

load_dotenv()

print(f"Pinecone version: {pinecone.__version__}")
print(f"PINECONE_API_KEY: {os.environ.get('PINECONE_API_KEY')[:5]}..." if os.environ.get('PINECONE_API_KEY') else "Not set")
print(f"PINECONE_ENV: {os.environ.get('PINECONE_ENV')}")
print(f"PINECONE_INDEX: {os.environ.get('PINECONE_INDEX')}")
print(f"PINECONE_HOST: {os.environ.get('PINECONE_HOST')}")

try:
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENV"]
    )
    print("Pinecone initialized successfully")

    index = pinecone.Index(os.environ["PINECONE_INDEX"])
    print(f"Successfully connected to index: {os.environ['PINECONE_INDEX']}")

    stats = index.describe_index_stats()
    print(f"Index stats: {stats}")
except Exception as e:
    print(f"An error occurred: {e}")