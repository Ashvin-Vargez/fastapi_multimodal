# chat_manager.py

from typing import Dict, List, Optional
from datetime import datetime,timezone

class ChatManager:
    def __init__(self):
        # Structure: {user_id: {session_id: [messages]}}
        self.chat_history: Dict[str, Dict[str, List[dict]]] = {}
        
    def add_message(self, user_id: str, session_id: str, role: str, content: any) -> None:
        """Add a message to the chat history."""
        if user_id not in self.chat_history:
            self.chat_history[user_id] = {}
            
        if session_id not in self.chat_history[user_id]:
            self.chat_history[user_id][session_id] = []

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.chat_history[user_id][session_id].append(message)

        
    
    def get_chat_history(self, user_id: str, session_id: str) -> List[dict]:
        """Get chat history for a specific user and session."""
        return self.chat_history.get(user_id, {}).get(session_id, [])
    
    def clear_chat_history(self, user_id: str, session_id: str) -> None:
        """Clear chat history for a specific user and session."""
        if user_id in self.chat_history and session_id in self.chat_history[user_id]:
            self.chat_history[user_id][session_id] = []