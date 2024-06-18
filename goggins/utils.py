import datetime


def extract_response_content(result):
    try:
        # Check if there is a valid response message
        if result.choices and result.choices[0].message and result.choices[0].message.tool_calls:
            for tool_call in result.choices[0].message.tool_calls:
                return tool_call.function.arguments
            return "Error: Something went wrong"
        else:
            return "No valid response received."
    except (IndexError, KeyError) as e:
        return f"Error: {str(e)}"
    
def get_current_date_time():
    return datetime.datetime.now().strftime("%Y-%m-%d")