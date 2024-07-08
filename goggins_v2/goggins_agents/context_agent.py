import chromadb, os
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

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
    img_embedding_function = OpenCLIPEmbeddingFunction()
    data_loader = ImageLoader()

    chromadb_client = chromadb.Client()
    collection = chromadb_client.get_collection(
        name="goggins_demo",
        embedding_function=img_embedding_function,
        data_loader=data_loader
    )
    number = min(top_results_num, collection.count())
    results = collection.query(
        query_texts=[query], 
        n_results=number,
        include=["uris", "documents", "distances"]
    )
    return results