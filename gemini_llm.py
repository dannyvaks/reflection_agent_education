import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
import google.generativeai as genai
from langgraph.graph import MessageGraph, END

# Load environment variables
load_dotenv()

# Configure Google Generative AI
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

genai.configure(api_key=GOOGLE_API_KEY)


class GenaiLLM:
    """
    A wrapper for the Google Generative AI model (Gemini)
    that provides a similar interface to the LangChain ChatOpenAI class.
    """

    def __init__(self, model_name="gemini-2.0-flash", temperature=0.2):
        """Initialize the Gemini model."""
        self.model_name = model_name
        self.temperature = temperature
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={"temperature": self.temperature}
        )

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """
        Process a list of messages and return an AIMessage response.

        Args:
            messages (List[BaseMessage]): A list of LangChain message objects.

        Returns:
            AIMessage: The model's response as an AIMessage.
        """
        # Convert LangChain messages to Gemini's format
        gemini_messages = []

        for message in messages:
            if isinstance(message, SystemMessage):
                # Gemini doesn't have a system role, so we'll add it as a user message
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": f"System instruction: {message.content}"}]
                })
            elif isinstance(message, HumanMessage):
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": message.content}]
                })
            elif isinstance(message, AIMessage):
                gemini_messages.append({
                    "role": "model",
                    "parts": [{"text": message.content}]
                })

        # If there's only one message and it's from the user, use generate_content directly
        if len(gemini_messages) == 1 and gemini_messages[0]["role"] == "user":
            try:
                response = self.model.generate_content(gemini_messages[0]["parts"][0]["text"])
                return AIMessage(content=response.text)
            except Exception as e:
                print(f"Error in direct generation: {e}")
                # If the input is too short, add some context
                try:
                    response = self.model.generate_content(
                        "Please respond to the following request: " + gemini_messages[0]["parts"][0]["text"]
                    )
                    return AIMessage(content=response.text)
                except Exception as e2:
                    print(f"Error in fallback generation: {e2}")
                    return AIMessage(content="I was unable to process that request.")

        # For multi-turn conversations, use chat
        try:
            chat = self.model.start_chat(history=[])

            # Add each message to the chat
            response = None
            for msg in gemini_messages:
                if msg["role"] == "user":
                    response = chat.send_message(msg["parts"][0]["text"])
                # We don't need to send model messages as they're just for context

            # Return the last response
            if response:
                return AIMessage(content=response.text)
            else:
                # If we somehow didn't get a response, create one
                response = self.model.generate_content("Please provide a helpful response.")
                return AIMessage(content=response.text)

        except Exception as e:
            print(f"Error in chat generation: {e}")
            # Fallback to simple generation with the last user message
            for msg in reversed(gemini_messages):
                if msg["role"] == "user":
                    try:
                        response = self.model.generate_content(
                            "Please respond to: " + msg["parts"][0]["text"]
                        )
                        return AIMessage(content=response.text)
                    except:
                        break

            return AIMessage(content="I was unable to process that conversation.")