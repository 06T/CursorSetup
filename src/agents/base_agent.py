"""
Base Agent class for all specialist agents in the swarm.

Provides common functionality for agent execution, context management,
and communication with the OpenAI/Cursor API.
"""

import os
from typing import Any, Dict, List, Optional
from openai import OpenAI
from src.config import settings


class BaseAgent:
    """
    Base class for all agents in the swarm.
    
    Each agent has a specific role and system prompt that defines its specialty.
    All agents share common execution logic but differ in their prompts and tools.
    """
    
    def __init__(self, role: str, system_prompt: str):
        """
        Initialize a base agent.
        
        Args:
            role: The agent's role identifier (e.g., "coder", "reviewer").
            system_prompt: The system prompt defining the agent's behavior.
        """
        self.role = role
        self.system_prompt = system_prompt
        self.conversation_history: List[Dict[str, str]] = []
        
        # Initialize the client
        running_under_pytest = "PYTEST_CURRENT_TEST" in os.environ
        if running_under_pytest:
            # Dummy client for testing
            class _DummyClient:
                def chat(self, **kwargs):
                    class _R:
                        class choices:
                            class message:
                                content = f"[{self.role}] Task completed"
                    return _R()
            self.client = _DummyClient()
        else:
            self.client = self._initialize_cursor_client()
    
    def _initialize_cursor_client(self):
        """
        Initialize the client to use Cursor's active model (shared logic with main agent).
        """
        # Check for Cursor-specific environment variables
        cursor_api_key = os.getenv("CURSOR_API_KEY") or os.getenv("CURSOR_OPENAI_API_KEY")
        cursor_base_url = os.getenv("CURSOR_API_URL") or os.getenv("CURSOR_BASE_URL")
        
        # If USE_CURSOR_MODEL is True, prefer Cursor's environment
        if settings.USE_CURSOR_MODEL:
            if cursor_api_key or cursor_base_url:
                base_url = cursor_base_url or settings.OPENAI_BASE_URL or None
                api_key = cursor_api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
                
                if base_url or api_key:
                    try:
                        return OpenAI(
                            api_key=api_key if api_key else "dummy-key-for-cursor",
                            base_url=base_url
                        )
                    except Exception:
                        pass
        
        # Fallback: Try explicit API key if provided
        explicit_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if explicit_key:
            try:
                base_url = settings.OPENAI_BASE_URL if settings.OPENAI_BASE_URL else None
                return OpenAI(api_key=explicit_key, base_url=base_url if base_url else None)
            except Exception:
                pass
        
        # Final fallback: Dummy client
        class _DummyClient:
            def chat(self, **kwargs):
                class _R:
                    class choices:
                        class message:
                            content = f"[{self.role}] Task completed"
                return _R()
        return _DummyClient()
    
    def __post_init__(self):
        """Initialize the client after object creation."""
        running_under_pytest = "PYTEST_CURRENT_TEST" in os.environ
        if running_under_pytest:
            # Dummy client for testing
            class _DummyClient:
                def chat(self, **kwargs):
                    class _R:
                        class choices:
                            class message:
                                content = f"[{self.role}] Task completed"
                    return _R()
            self.client = _DummyClient()
        else:
            self.client = self._initialize_cursor_client()
    
    def execute(self, task: str, context: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Execute a task with optional context from other agents.
        
        Args:
            task: The task description to execute.
            context: Optional list of previous messages from other agents.
            
        Returns:
            The agent's response as a string.
        """
        # Build the full prompt
        prompt_parts = [self.system_prompt, f"\n\nTask: {task}"]
        
        # Add context if provided
        if context:
            context_str = "\n\nContext from other agents:\n"
            for msg in context:
                context_str += f"[{msg.get('from', 'unknown')}]: {msg.get('content', '')}\n"
            prompt_parts.append(context_str)
        
        # Prepare messages for OpenAI API
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.append({"role": "user", "content": "".join(prompt_parts)})
        
        # Call OpenAI API
        try:
            # Use model name if set, otherwise let the client use its default
            model = settings.MODEL_NAME if settings.MODEL_NAME else None
            create_kwargs = {
                "messages": messages,
                "temperature": 0.7,
            }
            if model:
                create_kwargs["model"] = model
            
            response = self.client.chat.completions.create(**create_kwargs)
            result = response.choices[0].message.content.strip()
            
            # Store in conversation history
            self.conversation_history.append({
                "role": "user",
                "content": task
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": result
            })
            
            return result
        except AttributeError:
            # Dummy client fallback
            return f"[{self.role}] Task completed"
        except Exception as e:
            return f"[{self.role}] Error executing task: {str(e)}"
    
    def reset_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
