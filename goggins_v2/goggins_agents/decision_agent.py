import chromadb, os
from dotenv import load_dotenv
from goggins_agents.context_agent import context_agent
from openai import OpenAI

from utils import consolidate_text, consolidate_uris, encode_image

load_dotenv()

checkin_iteration = 0

tools = [
  {
    "type": "function",
    "function": {
        "name": "decisionUpdate",
        "description": """Call this function. If you think it is appropriate for the user to make an update on their progress to social media, include the message parameter.
                        If you think it is appropriate to charge the user for a monetary value (lack of progress), include the charge parameter.""",
        "parameters": {
            "type": "object",
            "properties": {
                "social_update": {
                    "type": "string",
                    "description": """Message to the user requesting them to post a social update that is related to the current task/goal if appropriate. 
                    NOT DIRECT MESSAGE FOR THE USER TO POST ON SOCIAL MEDIA, just an ask."""
                },
                "charge": {
                    "type": "string",
                    "description": "A dollar amount (in the format $xx.xx) that charges the user and keeps them accountable if the goals are not being met."
                }
            }
        }
    }
  }
]

def extract_decision_content(result):
    try:
        # Check if there is a valid response message
        if result.choices and result.choices[0].message and result.choices[0].message.tool_calls:
            for tool_call in result.choices[0].message.tool_calls:
                return tool_call.function.arguments
            print("Error: Something went wrong")
            return "Error: Something went wrong"
        else:
            print("No Social Media Post Today!")
            return "No Social Media Post Today!"
    except (IndexError, KeyError) as e:
        return f"Error: {str(e)}"

def decision_agent(current_tasks, response, opinion):
    if (len(current_tasks['tasks']) == 0):
        return """{"social_update": "Why don't you make a final Social Media post today? Congratulations on your incredible acheivement!"}"""
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    tmp = current_tasks['tasks'][0]
    description = tmp['task_description']
    completion_day = tmp['completion_day']
    headline = current_tasks['headline']
    context = context_agent(query=description, top_results_num=5)
    prompt = f'Determine whether it is appropriate for the user to make a social update, or be charged a monetary value for lack of progress.\n'
    if context:
        prompt += 'Take into account the previous information you\'ve gotten from the user:' + consolidate_text(context)
        prompt += '\nYou can also use the images given to you as context but JUST AS CONTEXT. \n'
    prompt += f'\nYour ultimate goal is to help the user: {headline}. Here is the current list of tasks to be done: {current_tasks} \n\n\n\n'
    prompt += f'Here is the user\'s current update {response}.\n\n'
    prompt += f"""Use the legit-agent's user reliability opinion: {opinion}. This opinion is quite important, let
                it influence the decision you make—epecially related to charging users."""

    user_content = [{"type": "text", "text": prompt}]
    if context and consolidate_uris(context):
        for image_path in consolidate_uris(context):
            base64_image = encode_image(image_path)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }
             })

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """You are part of Goggins AI, an AI agent that acts as an accountability buddy. You are the decision agent.
             Read the context from the user as well as dates you got that context. Given the task descriptions, determine whether a social update on the user's progress
             is reasonable and/or a monetary penalty. YOU MUST CALL THE decisionUpdate FUNCTION BUT DON'T HAVE TO INCLUDE ANY PARAMETERS IF AN UPDATE/CHARGE IS NOT NEEDED. AGAIN, DO NOT ALWAYS MAKE THE USER 
             POST A SOCIAL UPDATE OR PAY A FINE!!!!"""},
            {"role": "user", "content": user_content}
        ],
        tools=tools,
        tool_choice="auto"
    )
    return extract_decision_content(completion)