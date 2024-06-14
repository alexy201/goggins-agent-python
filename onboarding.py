import datetime, os, json
from goggins import after_onboard
from openai import OpenAI
from dotenv import load_dotenv

from utils import extract_response_content, get_current_date_time

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

system = f"You are an accountability buddy that is responsible for organizing a plan to achieve a goal for the user. The current date is {get_current_date_time()}. Please use the function to create a list of sub-tasks and the required completion date."

tools = [
  {
    "type": "function",
    "function": {
        "name": "createPlan",
        "description": "Call this function if the request asks to create or edit a plan to accomplish a user's goal",
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
                                "description": "The future day to complete this task in the format MM/DD/YYYY. DO NOT EVER RETURN ANY OTHER FORMAT!!!!)"
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

def construct_prompt(prompt, history, tasks):
    consolidate = "User's Previous Requests: <<<User>>>\n"
    for message in history:
        consolidate += f"{message}\n"
    consolidate += "<<</User>>>\n"
    
    ret = (
        f"Your Previous Plan: <<<PLAN>>>{tasks}<<</PLAN>>>\n\n\n" +
        consolidate +
        f"User's Current Request: <<<REQUEST>>>{prompt}<<</REQUEST>>>\n"
    )
    return ret


if __name__ == "__main__":
    prompt = input("Enter the goal you want to accomplish: ")
    history = []
    tasks = ""

    while (prompt != "Done"):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": construct_prompt(prompt, history, tasks)}
        ]
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        tasks = extract_response_content(completion)
        res = json.loads(tasks)
        print(json.dumps(res, indent = 4))
        history.append(prompt)
        prompt = input("How should I change the schedule? If you are happy with the schedule, just put \"Done\": ")

    after_onboard(tasks, history)

