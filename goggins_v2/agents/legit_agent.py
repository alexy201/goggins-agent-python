import chromadb, os, json, datetime
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
from agents.context_agent import context_agent
from agents.planning_agent import consolidate_context
from openai import OpenAI

load_dotenv()

def extract_verification_content(result):
    try:
        # Check if there is a valid response message
        if result.choices and result.choices[0].message and result.choices[0].message.tool_calls:
            for tool_call in result.choices[0].message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                if ('need_image' in args):
                    return args['need_image']
            return None
        else:
            return None
    except (IndexError, KeyError) as e:
        return f"Error: {str(e)}"
    
def validate_file(file_path):
    # Dummy implementation of validate_file
    return os.path.exists(file_path) and (
        file_path.endswith('.jpg') or file_path.endswith('.jpeg')
        or file_path.endswith('.JPEG') or file_path.endswith('.png')
        or file_path.endswith('.svg') or file_path.endswith('.PNG')
        or file_path.endswith('.JPG')
    )

def get_file_input(res):
    file_path = input(res + " ").strip().strip('"').strip("'")
    while True:
        if validate_file(file_path):
            return file_path
        else:
            file_path = input("Invalid file. Please try again.")

tools = [
    {
        "type": "function",
        "function": {
            "name": "getImageVerification",
            "description": "Call this function to request an image verification from the user, if you believe the user's update to be unrealistic or suspicious.",
            "parameters": {
                "type": "object",
                "properties": {
                    "need_image": {
                        "type": "string",
                        "description": "A prompt asking the user for image verification (also try to explain a little bit why you think the user is being suspicious)."
                    }
                },
                "required": ["need_image"]
            }
        }
    }
]

def legit_agent(current_tasks, response):

    ##### SET UP THE AGENT #####
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    tmp = current_tasks['tasks'][0]
    description = tmp['task_description']
    completion_day = tmp['completion_day']
    headline = current_tasks['headline']
    context = context_agent(query=description, top_results_num=5)

    ##### DETERMINE IF IMAGE IS NEEDED/GET FILE INPUT #####
    prompt = f"""Determine whether the user\'s update is suspicious or unreasonable. For example, if the user is being dodgy (short updates or unspecific updates), or if
    the user completes tasks too easily or quickly. You are the person responsible for making sure that the user is not being untruthful and staying accountable. \n"""
    if context:
        prompt += 'Take into account the previous information you\'ve gotten from the user:' + consolidate_context(context)
    prompt += f'\nYour ultimate goal is to help the user: {headline}. Here is the current list of tasks to be done: {current_tasks} \n\n\n\n'
    prompt += f'Here is the user\'s current update {response}.'
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"""You are part of Goggins AI, an AI agent that acts as an accountability buddy. You are the legit agent.
             Read the context from the user as well as dates you got that context. Given the user's progress, determine whether it is reasonable to ask for an 
             image verification. Remember, try asking for verifications if the user is being dodgy (short updates or unspecific updates), or if 
             the user completes tasks too easily or quickly. Make sure that the task is one where doing image verification is reasonable."""},
            {"role": "user", "content": prompt}
        ],
        tools=tools,
        tool_choice="auto"
    )
    res = extract_verification_content(completion)
    if (res):
        get_file_input(res)

    ##### GET OPINION #####
    



        