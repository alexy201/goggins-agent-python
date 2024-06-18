import base64
import datetime
import os

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def save_image(checkin_iteration, image, folder="images", image_name=None):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if image_name is None:
        image_name = f"user_image_{checkin_iteration}.jpg"
    image_path = os.path.join(folder, image_name)
    with open(image_path, "wb") as img_file:
        img_file.write(image)
    return image_path

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

def consolidate_descriptions(current_tasks):
    consolidated = ""
    for task in current_tasks['tasks']:
        consolidated += task['task_description'] + "\n"
    return consolidated

def consolidate_text(context):
    documents = context.get('documents', [])
    return '\n'.join(doc for doc_list in documents if doc_list for doc in doc_list if doc)

def consolidate_uris(context):
    uris = context.get('uris', [])
    filtered_uris = [uri for uri_list in uris if uri_list for uri in uri_list if uri]
    return filtered_uris
