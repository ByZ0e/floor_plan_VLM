import re
import base64
from openai import AzureOpenAI
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
deployment_name = "gpt-4o-2024-05-13"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def request_base64_gpt4v(message_content, system_content=None, seed=42):
    api_key=OPENAI_API_KEY
    azure_endpoint="https://yjzl-openai-zywang.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview"
    
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2025-04-01-preview",
        azure_endpoint=azure_endpoint
    )
    
    if not system_content:
        messages=[
            {
                "role": "user",
                "content": message_content,
            }
        ]
    else:
        messages=[
            {
                "role": "system",
                "content": system_content,

            },
            {
                "role": "user",
                "content": message_content,
            }
        ]

    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        max_tokens=4096,
        temperature=0.7,
        seed=seed
    )
    
    return response.choices[0].message.content


def request_gpt4(message_content):
    api_key=OPENAI_API_KEY
    azure_endpoint="https://yjzl-openai-zywang.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview"

    client = AzureOpenAI(
        api_key=api_key,
        api_version="2025-04-01-preview",
        azure_endpoint=azure_endpoint
    )
    response = client.chat.completions.create(
        model="gpt4",
        messages=[
            {
            "role": "user",
            "content": message_content,
            }
        ],
        max_tokens=1024,
        temperature=0,
        seed=42
    )
    
    return response.choices[0].message.content
