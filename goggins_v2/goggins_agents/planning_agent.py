import chromadb, os
from dotenv import load_dotenv
from openai import OpenAI
from goggins_agents.context_agent import context_agent
from utils import consolidate_text, consolidate_descriptions, consolidate_uris, encode_image, extract_response_content

load_dotenv()


tools = [
  {
    "type": "function",
    "function": {
        "name": "createPlan",
        "description": """Call this function if the request asks to create or edit a plan to accomplish a user's goal.
        Remember to delete tasks if the user has reasonably and successfully completed them.""",
        "parameters": {
            "type": "object",
            "properties": {
                "headline": {
                    "type": "string",
                    "description": "A good title/subject headline for the goal that is not too long."
                },
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task_description": {
                                "type": "string",
                                "description": "A description of the sub-task required to eventually achieve the main goal."
                            },
                            "completion_day": {
                                "type": "string",
                                "description": """The future day to complete this task in the format MM/DD/YYYY. DO NOT EVER RETURN ANY OTHER FORMAT!!!!
                                Again, all days must be in the future (NOT ON THE SAME DAY)."""
                            }
                        }
                    }
                }
            },
            "required": ["headline", "tasks"]
        }
    }
  }
]


def planning_agent(current_tasks, response, opinion):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    tmp = current_tasks['tasks'][0]
    description = tmp['task_description']
    completion_day = tmp['completion_day']
    headline = current_tasks['headline']
    context = context_agent(query=consolidate_descriptions(current_tasks), top_results_num=5)

    prompt = f'You are processing a check-in for the following task: {description}. The current day is: {completion_day}\n'
    if context:
        prompt += 'Take into account the previous information you\'ve gotten from the user:' + consolidate_text(context)
        prompt += '\nYou can also use the images given to you as context but JUST AS CONTEXT. \n'
    prompt += f'\nYour ultimate goal is to help the user: {headline}. Here is the current list of tasks to be done: {current_tasks} \n\n\n\n'
    prompt += f"""Here is the user current update {response}. Given the user update for the task, change/alter (or keep it the same) the user current plan. 
    Feel free to remove any tasks, change the timeline, or do anything reasonable with the schedule. If the user completed the task, remove it.\n\n"""
    prompt += f"""Use the legit-agent's user reliability opinion: {opinion}. This opinion is quite important, let
                it influence the progression of tasks/whether to remove tasks or not."""

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
            {"role": "system", "content": """
You are part of Goggins AI, an AI agent that acts as an accountability buddy. You are the planning agent. 
Read the context from the user as well as dates you got that context. Given the user update for the task, change/alter (or keep it the same) the user current plan.
Feel free to remove any tasks, change the timeline, or do anything reasonable with the schedule. If the user completed the task, remove it.
VERY IMPORTANT: MAKE SURE THE DATES FOR SUBSEQUENT TASKS ARE IN CHRONILOGICAL ORDER!"""},
            {"role": "user", "content": user_content}
        ],
        tools=tools,
        tool_choice="auto"
    )
    return extract_response_content(completion)
