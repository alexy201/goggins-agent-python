import chromadb, os, json, datetime
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
from agents.context_agent import context_agent
from agents.planning_agent import consolidate_context
from openai import OpenAI

load_dotenv()

checkin_iteration = 0

tools = [
  {
    "type": "function",
    "function": {
        "name": "socialMediaUpdate",
        "description": "Call this function. If you think it is appropriate for the user to make an update on their progress to social media, include the message parameter.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message to the user asking for an update that is related to the current task/goal if appropriate. NOT MESSAGE FOR THE USER TO POST."
                }
            }
        }
    }
  }
]

def extract_social_content(result):
    try:
        # Check if there is a valid response message
        if result.choices and result.choices[0].message and result.choices[0].message.tool_calls:
            for tool_call in result.choices[0].message.tool_calls:
                return tool_call.function.arguments
            return "Error: Something went wrong"
        else:
            return "No Social Media Post Today!"
    except (IndexError, KeyError) as e:
        return f"Error: {str(e)}"

def social_agent(current_tasks, response):
    if (len(current_tasks) == 0):
        return "Why don't you make a final Social Media post today? Congratulations on your incredible acheivement!"
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    tmp = current_tasks['tasks'][0]
    description = tmp['task_description']
    completion_day = tmp['completion_day']
    headline = current_tasks['headline']
    context = context_agent(query=description, top_results_num=5)
    prompt = f'Determine whether it is appropriate for the user to make a social update.\n'
    if context:
        prompt += 'Take into account the previous information you\'ve gotten from the user:' + consolidate_context(context)
    prompt += f'\nYour ultimate goal is to help the user: {headline}. Here is the current list of tasks to be done: {current_tasks} \n\n\n\n'
    prompt += f'Here is the user\'s current update {response}.'
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """You are part of Goggins AI, an AI agent that acts as an accountability buddy. You are the social media agent.
             Read the context from the user as well as dates you got that context. Given the task descriptions, determine whether a social update on the user's progress
             is reasonable. YOU MUST CALL THE socialMediaUpdate FUNCTION BUT DON'T HAVE TO INCLUDE ANY PARAMETERS IF AN UPDATE IS NOT NEEDED. AGAIN, DO NOT ALWAYS MAKE THE SURE 
             POST A SOCIAL UPDATE!!!!"""},
            {"role": "user", "content": prompt}
        ],
        tools=tools,
        tool_choice="auto"
    )
    return extract_social_content(completion)