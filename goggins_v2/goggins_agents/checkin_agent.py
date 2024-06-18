import os
from dotenv import load_dotenv
from goggins_agents.context_agent import context_agent
from openai import OpenAI

from utils import consolidate_text, consolidate_uris, encode_image

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
        prompt += 'Take into account the previous information you\'ve gotten from the user:' + consolidate_text(context)
        prompt += '\nYou can also use the images given to you as context but JUST AS CONTEXT. \n'
    prompt += f'\nYour ultimate goal is to help the user: {headline}\nCheck-in Message:'

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
            {"role": "system", "content": f"""You are part of Goggins AI, an AI agent that acts as an accountability buddy. You are the check-in agent.
             Read the context from the user as well as dates you got that context. Given the task description/completion day, check-in on the user's progress.
             Feel free to push the user (push them hard) or complement them whenever it makes sense. THE CHECK IN SHOULD ONLY BE ONE SENTENCE NOT A LONG REMINDER.
             Start each check-in with 'Today is {completion_day}: ' """},
            {"role": "user", "content": user_content}
        ]
    )
    message = completion.choices[0].message
    return message.content


