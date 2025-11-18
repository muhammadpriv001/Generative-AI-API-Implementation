# Libraries Imported
import sqlite3
import faiss
import numpy as np
import uuid
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from datetime import datetime
import os

# -----------------------------
# SQL Memory
# -----------------------------
class SQLMemory:
    def __init__(self, db_path="sql_memory.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.init_tables()

    def init_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                text TEXT,
                timestamp TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id TEXT PRIMARY KEY,
                summary TEXT,
                timestamp TEXT
            );
        """)
        self.conn.commit()

    def add_memory(self, mid, text):
        cur = self.conn.cursor()
        cur.execute("INSERT INTO memories (id, text, timestamp) VALUES (?, ?, ?)",
                    (mid, text, datetime.now().isoformat()))
        self.conn.commit()

    def fetch_all_memories(self):
        cur = self.conn.cursor()
        cur.execute("SELECT text FROM memories")
        return [row[0] for row in cur.fetchall()]

    def save_summary(self, sid, summary):
        cur = self.conn.cursor()
        cur.execute("INSERT INTO summaries (id, summary, timestamp) VALUES (?, ?, ?)",
                    (sid, summary, datetime.now().isoformat()))
        self.conn.commit()

# -----------------------------
# FAISS Vector Memory
# -----------------------------
class VectorMemory:
    def __init__(self, dim=256, persist_path="vectors/faiss.index"):
        self.dim = dim
        self.persist_path = persist_path
        self.index = faiss.IndexFlatL2(dim)
        self.vectors = {}
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)

    def embed(self, text):
        np.random.seed(abs(hash(text)) % 100000)
        return np.random.rand(self.dim).astype('float32')

    def add(self, mid, text):
        vec = self.embed(text)
        self.vectors[mid] = vec
        self.index.add(np.array([vec]))

    def search(self, query, k=5):
        if not self.vectors:
            return [], []
        q = self.embed(query)
        dists, idxs = self.index.search(np.array([q]), k)
        return dists[0], idxs[0]

# -----------------------------
# Memory Manager using LangGraph StateGraph
# -----------------------------
class MemoryManager:
    def __init__(self):
        self.sql = SQLMemory()
        self.faiss = VectorMemory()
        self.graph = self.build_graph()

    def build_graph(self):
        # TypedDict for graph state
        class MemState(TypedDict):
            input_text: str
            facts_to_sql: list[str]
            facts_to_faiss: list[str]

        builder = StateGraph(MemState)

        # Node: Extract facts
        def extract(state: MemState) -> MemState:
            text = state['input_text']
            facts = [f.strip() for f in text.split('. ') if f.strip()]
            return {'facts_to_sql': facts, 'facts_to_faiss': facts}

        # Node: Decide what to store (currently all facts stored)
        def decide(state: MemState) -> MemState:
            return state

        # Node: Write to SQL (only new keys, don't touch input_text)
        def write_sql(state: MemState) -> MemState:
            for f in state.get('facts_to_sql', []):
                mid = str(uuid.uuid4())
                self.sql.add_memory(mid, f)
            return {}  # return empty dict to avoid overwriting input_text

        # Node: Write to FAISS
        def write_faiss(state: MemState) -> MemState:
            for f in state.get('facts_to_faiss', []):
                mid = str(uuid.uuid4())
                self.faiss.add(mid, f)
            return {}  # return empty dict

        # Add nodes
        builder.add_node('extract', extract)
        builder.add_node('decide', decide)
        builder.add_node('write_sql', write_sql)
        builder.add_node('write_faiss', write_faiss)

        # Connect nodes
        builder.add_edge(START, 'extract')
        builder.add_edge('extract', 'decide')
        builder.add_edge('decide', 'write_sql')
        builder.add_edge('decide', 'write_faiss')
        builder.add_edge('write_sql', END)
        builder.add_edge('write_faiss', END)

        return builder.compile()

    def add_memory(self, text: str):
        """Add a new piece of memory through LangGraph."""
        self.graph.invoke({'input_text': text})

    def run_weekly_summary(self):
        """Summarize SQL memories weekly using LLM."""
        from chatBot import client  # import client dynamically

        texts = self.sql.fetch_all_memories()
        if not texts:
            print("No memories to summarize.")
            return

        # Combine text for summarization
        combined_text = " | ".join(texts)[:2000]  # limit chars for safety

        # Call LLM to generate a proper summary
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=f"Summarize the following memories into a short summary:\n{combined_text}"
        )

        # Save summary in SQL
        sid = str(uuid.uuid4())
        self.sql.save_summary(sid, response.text)
        print(f"Weekly memory summary saved at {datetime.now().isoformat()}")