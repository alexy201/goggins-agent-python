import datetime
import chromadb, os, json
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv

from agents.checkin_agent import process_user_agent, prompt_user_agent
from agents.planning_agent import planning_agent
from agents.decision_agent import decision_agent
from utils import get_current_date_time

load_dotenv()

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=os.getenv("EMBEDDING_MODEL")
)

chromadb_client = chromadb.PersistentClient(path="chroma/")
collection = chromadb_client.create_collection(
    name="goggins_demo",
    embedding_function=openai_ef
)
current_tasks = []

def after_onboard(tasks, history):
    print("---------------------------------------------------------------------------")
    print("-----------------------------ONBOARDING DONE!------------------------------")
    print("---------------------------------------------------------------------------")

    onboard = f"User's Onboarding Messages {get_current_date_time()}: "
    for hist in history:
        onboard += f"\n\n\"{hist}\"\n\n"
    collection.add(
        documents=[onboard],
        ids=["onboard"]
    )
    global current_tasks
    current_tasks = json.loads(tasks)
    print("CURRENT TASKS: \n" + json.dumps(current_tasks, indent=4))
    start_goggins()


def start_goggins():
    global current_tasks
    print("---------------------------------------------------------------------------")
    print("----------------------------Starting Goggins AI----------------------------")
    print("---------------------------------------------------------------------------")
    while (len(current_tasks['tasks']) > 0):
        agent_prompt = prompt_user_agent(current_tasks)
        checkin = input(agent_prompt)
        process_user_agent(agent_prompt, checkin, current_tasks)
        tmp = planning_agent(current_tasks, checkin)
        current_tasks = json.loads(tmp)
        print("----------------------------NEW TASK PLAN----------------------------")
        print(json.dumps(current_tasks, indent=4))
        decision = json.loads(decision_agent(current_tasks, checkin))
        print(f"AGENT DECISION: {json.dumps(decision, indent=4)}")
    print("CONGRATULATIONS! YOU'VE COMPLETED THE GOAL")