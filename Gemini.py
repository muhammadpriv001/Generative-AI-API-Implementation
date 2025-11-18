# Libraries Imported

import os
from dotenv import load_dotenv
import cv2
from google import genai
from google.genai import types
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
import threading
from PIL import Image

# Set up Gemini API

load_dotenv()
api_key = os.getenv("API_KEY")

client = genai.Client(api_key=api_key)

# Initialize conversation memory
memory = ConversationBufferMemory()

# Add a primary history to the memory
primary_history = """
The bot is a helpful assistant named JARVIS short for Just A Rather Very Intelligent System. It can assist with various queries, provide guidance, and respond politely.
"""
memory.save_context({"input": ""}, {"output": primary_history})

# Function to perform text completion

def text_completion(prompt):
    # Update conversation memory
    conversation_history = memory.buffer
    conversation_history = conversation_history.replace("AI:", "")

    # Generate new response based on updated prompt
    response = client.models.generate_content(
        model="gemini-2.5-pro", contents=f"{conversation_history} {prompt}"
    )
    
    # Store new conversation context in memory
    memory.save_context({"input": prompt}, {"output": response.text})
        
    return response.text

# Function to describe video feed

def describe_video_feed(query, frame):
    # Retrieve the most recent conversation from memory buffer
    conversation_history = memory.buffer
    conversation_history = conversation_history.replace("AI:", "")
    
    # Save and open the frame as an image
    cv2.imwrite("D:\\Programming\\Projects\\Desktop Applications\\Generative-AI-API-Implementation\\img.jpg", frame)
    img = client.files.upload(file="D:\\Programming\\Projects\\Desktop Applications\\Generative-AI-API-Implementation\\img.jpg")
    
    try:
        # Send the query and frame to the model for analysis
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=[img, f"{conversation_history} {query}"]
        )
        # Save each response and context
        memory.save_context({"input": query}, {"output": response.text})
        return response.text
    except Exception as e:
        return f"Failed to describe image: {e}"

# Function to handle video feed

def video_feed(thread_event):
    global frame
    cap = cv2.VideoCapture(0)  # Start video capture from webcam

    while not thread_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Cam", frame)

        # Press 'q' to exit the live feed
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            thread_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to handle text input and process user queries

def text_input(thread_event):
    keyword = "video feed"

    while not thread_event.is_set():
        query = input("Enter your query or type 'exit' to quit: \nUser:")
        if query.lower() == 'exit':
            thread_event.set()
            break
        elif keyword in query.lower():
            query = query.replace("video feed", "").strip()
            description = describe_video_feed(query, frame)
            print(f"J.A.R.V.I.S.: {description}")
            os.remove("D:\\Programming\\Projects\\Desktop Applications\\Generative-AI-API-Implementation\\img.jpg")
        else:
            response = text_completion(query)
            print(f"J.A.R.V.I.S.: {response}")

# Main Function

if __name__ == "__main__":
    # Event to signal threads to stop
    
    thread_event = threading.Event()

    # Create threads for video feed and text input
    
    video_thread = threading.Thread(target=video_feed, args=(thread_event,))
    text_thread = threading.Thread(target=text_input, args=(thread_event,))

    # Start the threads
    
    video_thread.start()
    text_thread.start()

    # Ensure both threads finish before exiting
    
    video_thread.join()
    text_thread.join()