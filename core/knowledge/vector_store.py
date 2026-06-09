# core/knowledge/vector_store.py
import sqlite3
import numpy as np

class SQLiteVectorStore:
    def __init__(self, db_path: str = "factory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    main_idea TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id INTEGER,
                    text_content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    syntax_type TEXT,
                    FOREIGN KEY(agent_id) REFERENCES agents(id)
                )
            """)
            conn.commit()

    def register_agent(self, name: str, idea: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO agents (name, main_idea) VALUES (?, ?)", 
                    (name, idea)
                )
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                cursor.execute(
                    "SELECT id FROM agents WHERE name = ?", (name,)
                )
                return cursor.fetchone()[0]

    def clear_agent_knowledge(self, agent_id: int):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM knowledge_chunks WHERE agent_id = ?", 
                (agent_id,)
            )
            conn.commit()

    def add_chunks(self, agent_id: int, chunks: list, embeddings: np.ndarray):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for idx, chunk in enumerate(chunks):
                vector_1d = embeddings[idx].flatten().astype(np.float32)
                vec_blob = vector_1d.tobytes()
                cursor.execute("""
                    INSERT INTO knowledge_chunks (agent_id, text_content, embedding, syntax_type)
                    VALUES (?, ?, ?, ?)
                """, (agent_id, chunk["content"], vec_blob, chunk["syntax_type"]))
            conn.commit()

    def search_top_k(self, agent_id: int, query_vector: np.ndarray, k: int = 2) -> list:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, text_content, embedding FROM knowledge_chunks "
                "WHERE agent_id = ?", (agent_id,)
            )
            rows = cursor.fetchall()

        if not rows:
            return []

        db_vectors = []
        db_metadata = []

        for row_id, text, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.shape[0] == 768:
                db_vectors.append(vec)
                db_metadata.append({"id": row_id, "text": text})

        if not db_vectors:
            return []

        matrix = np.array(db_vectors)
        flat_query = query_vector.flatten()
        
        dot_product = np.dot(matrix, flat_query)
        matrix_norms = np.linalg.norm(matrix, axis=1)
        query_norm = np.linalg.norm(flat_query)
        
        scores = dot_product / (matrix_norms * query_norm + 1e-9)
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append({
                "text": db_metadata[idx]["text"],
                "score": float(scores[idx])
            })
        return results[0] if results else []
