# gpt4_o.py

from typing import List
from dotenv import load_dotenv
import os
import requests
from fastapi import HTTPException

# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

MAX_IMAGES_PER_REQUEST = 100  # Maximum number of images per request

async def analyze_with_gpt4o(
    base64_images: List[str],
    prompt: str,
    chat_history: List[dict]
) -> dict:
    """
    Analyze multiple images using GPT-4o model with chat history
    
    Args:
        base64_images (List[str]): List of base64 encoded images
        prompt (str): User's prompt/question
        chat_history (List[dict]): Previous chat messages
        
    Returns:
        dict: Analysis results
    """
    # Check if we exceed the maximum number of images
    if len(base64_images) > MAX_IMAGES_PER_REQUEST:
        raise ValueError(f"Maximum number of images per request is {MAX_IMAGES_PER_REQUEST}")
    
    # Construct the API messages
    api_messages = [
        {
            "role": "system",
            "content": """You are an AI assistant analyzing images and engaging in conversation about them. 
                        Answer questions based only on the images and text present in them. 
                        If multiple images are provided, consider all of them in your analysis.
                        If, no images are given, just answer the quetion. Give relevant and concise information."""
        }
    ]
    
    # Add chat history
    for msg in chat_history:
        # Only add text messages from history
        if isinstance(msg["content"], str):
            api_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    # Construct the content array with the prompt and images
    current_content = [{"type": "text", "text": prompt}]
    
    # Add images if present
    for idx, base64_image in enumerate(base64_images):
        current_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        })
    
    # Add the current message
    api_messages.append({
        "role": "user",
        "content": current_content
    })
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": "gpt-4o",
        "messages": api_messages,
        "max_tokens": 5000,
        "temperature": 0
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        return {
            "status": "success",
            "analysis": response.json()["choices"][0]["message"]["content"]
        }
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making request to OpenAI API: {str(e)}"
        )
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error parsing API response: {str(e)}"
        )