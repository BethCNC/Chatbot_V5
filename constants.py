import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")