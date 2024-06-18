import chromadb, os, json
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv

load_dotenv()

def context_agent(query: str, top_results_num: int):
    """
    Retrieves context for a given query from an index of tasks.

    Args:
        query (str): The query or objective for retrieving context.
        top_results_num (int): The number of top results to retrieve.

    Returns:
        list: A list of tasks as context for the given query, sorted by relevance.

    """
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=os.getenv("EMBEDDING_MODEL")
    )

    chromadb_client = chromadb.PersistentClient(path="chroma/")
    collection = chromadb_client.get_collection(
        name="goggins_demo",
        embedding_function=openai_ef
    )
    number = min(top_results_num, collection.count())
    results = collection.query(query_texts=[query], n_results=number)
    return results