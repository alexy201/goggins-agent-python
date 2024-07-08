import base64
import chromadb, os
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

from utils import get_current_date_time, save_image

load_dotenv()

checkin_iteration = 0

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def process_user_agent(agent_prompt, response, current_tasks, opinion, image):
    tmp = current_tasks['tasks'][0]
    description = tmp['task_description']
    completion_day = tmp['completion_day']

    global checkin_iteration
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
    # should be get_current_date_time() realistically
    message = f"User's Activity for task \"{description}\" on {completion_day}: {agent_prompt} {response}"
    opinion = f"Agent's Opinion for user response on {completion_day}: {opinion}"

    collection.add(
        ids=[f"user_message_{checkin_iteration}", f"agent_opinion_{checkin_iteration}"],
        documents=[message, opinion]
    )
    if image:
        collection.add(
            ids=[f"user_image_{checkin_iteration}"],
            uris=[f"{image}"]
        )

    checkin_iteration += 1
        


def onboard_context(history):
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=os.getenv("EMBEDDING_MODEL")
    )
    img_embedding_function = OpenCLIPEmbeddingFunction()
    data_loader = ImageLoader()

    chromadb_client = chromadb.Client()
    collection = chromadb_client.create_collection(
        name="goggins_demo",
        embedding_function=img_embedding_function,
        data_loader=data_loader
    )
    
    onboard = f"User's Onboarding Messages {get_current_date_time()}: "
    for hist in history:
        onboard += f"\n\n\"{hist}\"\n\n"
    collection.add(
        documents=[onboard],
        ids=["onboard"]
    )