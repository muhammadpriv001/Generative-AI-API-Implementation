# -----------------------------
# Libraries Imported
# -----------------------------
import os
import psutil
import time
from datetime import datetime
from dotenv import load_dotenv
import cv2
from google import genai
import threading
from memory_manager import MemoryManager

# -----------------------------
# Set up Gemini API
# -----------------------------
load_dotenv()
api_key = os.getenv("API_KEY")
client = genai.Client(api_key=api_key)

# -----------------------------
# Initialize Memory Manager
# -----------------------------
memory_manager = MemoryManager()

# -----------------------------
# Identity Handling
# -----------------------------
def get_user_identity():
    """Check SQL memory for stored user name."""
    memories = memory_manager.sql.fetch_all_memories()
    for m in memories:
        if m.lower().startswith("my name is"):
            return m.split("is")[-1].strip()
    return None

def initialize_identity():
    """Ask for name only if not already stored."""
    user_name = get_user_identity()
    if user_name:
        print(f"Welcome back, {user_name}!\n")
        return user_name

    # Ask once if missing
    print("Hello! Let's set up your profile.")
    user_name = input("Please enter your name: ").strip()
    if user_name:
        memory_manager.add_memory(f"My name is {user_name}")
        print(f"Profile saved. Hello {user_name}!\n")
    return user_name

# -----------------------------
# Weekly Summary Background Thread
# -----------------------------
def weekly_summary_scheduler():
    """
    Runs every hour, checks if today is Saturday.
    If Saturday, run weekly summary once.
    """
    summary_done = False

    while True:
        now = datetime.now()
        if now.weekday() == 5:  # Saturday = 5
            if not summary_done:
                try:
                    memory_manager.run_weekly_summary()
                except Exception as e:
                    print(f"Weekly summary failed: {e}")
                summary_done = True
        else:
            summary_done = False  # reset for next Saturday

        # Sleep for 1 hour (3600 sec). Adjust to smaller number for testing
        time.sleep(3600)

# -----------------------------
# Laptop Battery
# -----------------------------
def check_battery():
    battery = psutil.sensors_battery()
    if battery is None:
        return None  # Could be a desktop or battery info unavailable
    percent = battery.percent
    plugged = battery.power_plugged
    return percent, plugged

def battery_monitor():
    """
    Checks battery every minute.
    If battery < 3% and not charging, triggers AI farewell.
    """
    low_battery_alerted = False

    while True:
        status = check_battery()
        if status is None:
            break  # no battery info, exit thread

        percent, plugged = status
        if percent <= 3 and not plugged and not low_battery_alerted:
            # Low battery detected, prompt AI
            farewell = text_completion(
                "Laptop battery is critically low (<3%). Say goodbye to the user and see them off politely."
            )
            print(f"J.A.R.V.I.S.: {farewell}")
            low_battery_alerted = True

        # Reset flag if battery is okay or charging
        if percent > 5 or plugged:
            low_battery_alerted = False

        time.sleep(60)

# -----------------------------
# Text Completion Function
# -----------------------------
def text_completion(prompt):
    """Generate LLM response with SQL + FAISS context."""
    # Safe FAISS retrieval
    context_list = []
    if memory_manager.faiss.vectors:
        dists, idxs = memory_manager.faiss.search(prompt, k=5)
        keys = list(memory_manager.faiss.vectors.keys())
        context_list = [keys[i] for i in idxs if i < len(keys)]

    context = ' | '.join(context_list)
    sql_context = ' | '.join(memory_manager.sql.fetch_all_memories())

    full_context = f"User Facts: {sql_context}\nRelevant Memories: {context}\nUser: {prompt}"

    # Call Gemini LLM
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=full_context
            )
            break  # success
        except Exception as e:
            print(f"Gemini API busy, retrying ({attempt+1}/{max_retries})...")
            time.sleep(5)  # wait 5 seconds before retry
    else:
        return "J.A.R.V.I.S.: Sorry, Gemini API is busy. Please try again later."

    # Store prompt + response
    memory_manager.add_memory(prompt)
    memory_manager.add_memory(response.text)
    return response.text

# -----------------------------
# Video Feed Description
# -----------------------------
def describe_video_feed(query, frame):
    sql_context = ' | '.join(memory_manager.sql.fetch_all_memories())

    # Save frame temporarily
    temp_path = "temp_img.jpg"
    cv2.imwrite(temp_path, frame)
    img = client.files.upload(file=temp_path)

    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=[img, f"{sql_context} | {query}"]
        )
        memory_manager.add_memory(query)
        memory_manager.add_memory(response.text)
        os.remove(temp_path)
        return response.text
    except Exception as e:
        os.remove(temp_path)
        return f"Failed to describe image: {e}"

# -----------------------------
# Video Feed Thread
# -----------------------------
def video_feed(thread_event):
    global frame
    cap = cv2.VideoCapture(0)

    while not thread_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Cam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            thread_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# Text Input Thread
# -----------------------------
def text_input(thread_event):
    keyword = "video feed"

    while not thread_event.is_set():
        query = input("Enter your query or type 'exit' to quit:\nUser: ").strip()
        if 'exit' in query.lower():
            response = text_completion(
                f"{query}, tell the user that you are shutting down and at their disposal anytime they need!"
            )
            print(f"J.A.R.V.I.S.: {response}")
            thread_event.set()
            break
        elif "what is my name" in query.lower():
            user_name = get_user_identity()
            if user_name:
                print(f"J.A.R.V.I.S.: Your name is {user_name}")
            else:
                print("J.A.R.V.I.S.: I don't know your name yet.")
        elif keyword in query.lower():
            query_clean = query.replace("video feed", "").strip()
            description = describe_video_feed(query_clean, frame)
            print(f"J.A.R.V.I.S.: {description}")
        else:
            response = text_completion(query)
            print(f"J.A.R.V.I.S.: {response}")

# -----------------------------
# Main Function
# -----------------------------
if __name__ == "__main__":
    thread_event = threading.Event()

    # Initialize user identity once
    initialize_identity()

    # Start threads
    video_thread = threading.Thread(target=video_feed, args=(thread_event,))
    text_thread = threading.Thread(target=text_input, args=(thread_event,))
    summary_thread = threading.Thread(target=weekly_summary_scheduler, daemon=True)
    battery_thread = threading.Thread(target=battery_monitor, daemon=True)

    video_thread.start()
    text_thread.start()
    summary_thread.start()
    battery_thread.start()

    video_thread.join()
    text_thread.join()