import chromadb, os, json, datetime
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
from agents.context_agent import context_agent
from agents.planning_agent import consolidate_context
from openai import OpenAI

load_dotenv()

checkin_iteration = 0

def prompt_user_agent(current_tasks):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    tmp = current_tasks['tasks'][0]
    description = tmp['task_description']
    completion_day = tmp['completion_day']
    headline = current_tasks['headline']
    context = context_agent(query=description, top_results_num=5)
    prompt = f'Come up with a check-in for the following task: {description}. The current day is: {completion_day}\n'
    if context:
        prompt += 'Take into account the previous information you\'ve gotten from the user:' + consolidate_context(context)
    prompt += f'\nYour ultimate goal is to help the user: {headline}\nCheck-in Message:'
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"""You are part of Goggins AI, an AI agent that acts as an accountability buddy. You are the check-in agent.
             Read the context from the user as well as dates you got that context. Given the task description/completion day, check-in on the user's progress.
             Feel free to push the user (push them hard) or complement them whenever it makes sense. THE CHECK IN SHOULD ONLY BE ONE SENTENCE NOT A LONG REMINDER.
             Start each check-in with 'Today is {completion_day}: ' """},
            {"role": "user", "content": prompt}
        ]
    )
    message = completion.choices[0].message
    return message.content


def process_user_agent(agent_prompt, response, current_tasks):
    tmp = current_tasks['tasks'][0]
    description = tmp['task_description']
    completion_day = tmp['completion_day']

    global checkin_iteration
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=os.getenv("EMBEDDING_MODEL")
    )

    chromadb_client = chromadb.PersistentClient(path="chroma/")
    collection = chromadb_client.get_collection(
        name="goggins_demo",
        embedding_function=openai_ef
    )
    # should be get_current_date_time() realistically
    message = f"User's Activity for task \"{description}\" on {completion_day}: {agent_prompt} {response}"
    collection.add(
        documents=[message],
        ids=[f"user_message_{checkin_iteration}"]
    )
    checkin_iteration += 1

