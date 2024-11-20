# main.py

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional
from chat_manager import ChatManager
from file_processor import process_file
from gpt4_vision_handler import analyze_with_gpt4_vision

app = FastAPI()
chat_manager = ChatManager()

@app.post("/chat")
async def chat_endpoint(
    user_id: str = Form(...),
    session_id: str = Form(...),
    prompt: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """
    Chat endpoint that handles both text and file inputs
    
    Args:
        user_id (str): User identifier
        session_id (str): Session identifier
        prompt (str): User's message/prompt
        file (UploadFile, optional): Image or PDF file
        
    Returns:
        dict: Chat response with analysis
    """
    try:
        # Process file if provided
        base64_images = await process_file(file)
        
        # Get chat history
        chat_history = chat_manager.get_chat_history(user_id, session_id)
        
        # Add user message to history
        chat_manager.add_message(user_id, session_id, "user", prompt)
        
        # Get response from GPT-4 Vision
        response = await analyze_with_gpt4_vision(
            base64_images=base64_images,
            prompt=prompt,
            chat_history=chat_history
        )
        
        # Add assistant's response to history
        chat_manager.add_message(
            user_id,
            session_id,
            "assistant",
            response["analysis"]
        )
        
        return {
            "status": "success",
            "message": response["analysis"],
            "chat_history": chat_manager.get_chat_history(user_id, session_id)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional endpoint to clear chat history
@app.post("/clear-chat")
async def clear_chat(user_id: str, session_id: str):
    """Clear chat history for a specific user and session."""
    chat_manager.clear_chat_history(user_id, session_id)
    return {"status": "success", "message": "Chat history cleared"}