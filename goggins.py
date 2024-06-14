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
    print("--------------------------------ONBOARDING!--------------------------------")
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

    
if __name__ == "__main__":
    tasks = """{
    "headline": "Improve Social and Dating Skills by December",
    "tasks": [
        {
            "task_description": "Start a consistent workout routine (3 times a week). Eat a balanced diet. Regular grooming and hygiene. Update wardrobe if needed.",
            "completion_day": "06/30/2024"
        },
        {
            "task_description": "Attend a public speaking or confidence-building workshop. Practice daily affirmations and mindfulness.",
            "completion_day": "07/15/2024"
        },
        {
            "task_description": "Practice active listening techniques with friends and family. Engage in conversations with new people regularly. Understand and practice positive body language.",
            "completion_day": "07/31/2024"
        },
        {
            "task_description": "Join and participate in meetup groups or clubs related to your interests. Attend social gatherings and events. Join online dating apps and focus on engaging with people.",
            "completion_day": "08/31/2024"
        },
        {
            "task_description": "Practice asking people out on dates. Go on at least 2-3 dates to practice and gain experience. Reflect on each date to understand what went well and what could be improved.",
            "completion_day": "09/30/2024"
        },
        {
            "task_description": "Attend a social or dating workshop. Discuss experiences and get feedback from peers or advisors.",
            "completion_day": "10/15/2024"
        },
        {
            "task_description": "Plan and go on dates. Aim for at least one date every two weeks. Reflect and note areas of improvement after each date.",
            "completion_day": "10/30/2024"
        },
        {
            "task_description": "Review progress and identify any challenges faced. Adjust strategies as necessary.",
            "completion_day": "11/15/2024"
        },
        {
            "task_description": "Continue attending social gatherings and using dating apps. Aim to go on additional dates and apply learned techniques.",
            "completion_day": "11/30/2024"
        },
        {
            "task_description": "Evaluate overall progress. Celebrate successes and plan further steps to maintain and improve social/dating skills.",
            "completion_day": "12/15/2024"
        }
    ]
}
"""
    history = ['I want to get laid by december', 'can you put more events and check-ins on there, make it more specific']
    after_onboard(tasks, history)
    start_goggins()