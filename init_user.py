# Libraries Imported
import uuid

# Importing MemoryManager class
from memory_manager import MemoryManager

memory_manager = MemoryManager()

# Initialize user identity
user_facts = [
    "User name: Muhammad",
    "Favorite food: Biryani",
    "Profession: Software Engineer",
    "Interests: Robotics, AI, Programming"
]

for fact in user_facts:
    memory_manager.sql.add_memory(str(uuid.uuid4()), fact)
    memory_manager.faiss.add(str(uuid.uuid4()), fact)

print("✅ User identity initialized and stored in SQL + FAISS.")