import os
import pinecone
from dotenv import load_dotenv

def print_env_var(var_name):
    value = os.environ.get(var_name)
    if value:
        print(f"{var_name}: {value[:5]}..." if var_name == "PINECONE_API_KEY" else f"{var_name}: {value}")
    else:
        print(f"{var_name}: Not set")

# Load environment variables from .env
load_dotenv()

# Print the current Pinecone configuration for debugging
print(f"Pinecone version: {pinecone.__version__}")
print_env_var("PINECONE_API_KEY")
print_env_var("PINECONE_ENVIRONMENT")
print_env_var("PINECONE_INDEX")

try:
    # Initialize Pinecone
    pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    print("Pinecone initialized successfully")

    # List all available indexes
    print("Available indexes:")
    indexes = pc.list_indexes()
    print(indexes.names())

    index_name = os.environ["PINECONE_INDEX"]
    if index_name not in indexes.names():
        print(f"Warning: The specified index '{index_name}' is not in the list of available indexes.")

    # Connect to the Pinecone index
    index = pc.Index(index_name)
    print(f"Successfully connected to index: {index_name}")

    # Describe the index stats
    stats = index.describe_index_stats()
    print(f"Index stats: {stats}")

    # Test a simple query
    vector_dimension = stats['dimension']
    test_vector = [0.1] * vector_dimension
    query_result = index.query(vector=test_vector, top_k=1, include_metadata=True)
    print("Query test successful")
    print(f"Query result: {query_result}")

except KeyError as e:
    print(f"Missing environment variable: {e}. Please check your .env file.")
except Exception as e:
    print(f"An error occurred: {e}")

print("\nAdditional debugging information:")
print(f"Python version: {os.sys.version}")
print(f"Operating System: {os.name}")