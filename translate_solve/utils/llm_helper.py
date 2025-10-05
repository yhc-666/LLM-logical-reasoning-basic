"""
LLM Helper Module for Adaptive Solver
Provides simple interface for LLM calls using existing infrastructure
"""

import os
import sys
import json
from typing import Dict, Optional

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from openai import OpenAI


class LLMHelper:
    """Simple wrapper for LLM API calls"""
    
    def __init__(self, api_key: str, base_url: str, model: str = "gpt-4", 
                 temperature: float = 0.0, max_tokens: int = 4000):
        """
        Initialize LLM helper
        
        Args:
            api_key: OpenAI API key
            base_url: OpenAI API base URL
            model: Model name
            temperature: Temperature for generation
            max_tokens: Maximum tokens for generation
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(self, system_prompt: str, user_prompt: str, 
                 return_json: bool = False) -> str:
        """
        Generate response from LLM
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            return_json: Whether to parse response as JSON
            
        Returns:
            Response string or parsed JSON object
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            content = response.choices[0].message.content
            
            if return_json:
                # Try to extract JSON from the response
                # Handle cases where response might have extra text
                try:
                    # First try direct parsing
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Try to find JSON in the content
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    else:
                        # Return as dict with content
                        return {"response": content}
            
            return content
            
        except Exception as e:
            print(f"Error in LLM generation: {e}")
            raise