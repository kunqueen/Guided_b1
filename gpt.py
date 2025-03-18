import time
from http import HTTPStatus
import requests
import json
import base64
from lmdeploy.serve.openai.api_client import APIClient

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def query_qwen25(messages, temperature=0., max_tokens=1500):
    intern_messages = []
    for message in messages:
        intern_content = []
        for content in message["content"]:
            if "text" in content:
                intern_content.append({
                    "type": "text",
                    "text": content["text"]
                })
        intern_messages.append({
            "role": message["role"],
            "content": intern_content
        })

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "Qwen2_5-72B",
        "messages": intern_messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    response = requests.post("http://10.90.86.76:6008/v1/chat/completions", headers=headers, json=payload)
    if response.status_code != HTTPStatus.OK:
        raise ValueError('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
    response_output = response.json()
    role, content = response_output['choices'][0]['message']['role'], response_output['choices'][0]['message']['content']

    return role, content

if __name__ == '__main__':

    messages = []
    messages.append({
        "role": "system",
        "content": [{'text': "You are an assisst to help me generate the prompt for Diffusion Model to generate image."}]
    })
    messages.append({
            'role': 'user',
            'content': [
                {
                    'text': 
    """
    ### Task
    I have a violation description that is broken down into multiple levels or stages. 
    Please understand the reason for this violation, and then generate a prompt for Stable Diffusion that will create an image based on this violation. The prompt should instruct the model to generate an image that is a direct result of this violation.

    ### Output
    <prompt>

    ### Rule
    Finance Related->Financial Products->Cryptocurrency->Cryptocurrency Logos->Initial Coin Offering (ICO) Logos->Project Logotypes
    """
        },
    ]
    })

    role, content = query_qwen25(messages, temperature=0., max_tokens=1500)
    print(content)
    
