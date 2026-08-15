"""FAISS face/voice embedding store."""

import faiss
import numpy as np
import pickle
from pathlib import Path

class BiometricStore:
    FACE_DIM = 512
    VOICE_DIM = 256
    THRESHOLD_FACE = 0.6 #cosine similarity
    THRESHOLD_VOICE = 0.75

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.face_index = faiss.IndexFlatIP(self.FACE_DIM)
        self.voice_index = faiss.IndexFlatIP(self.VOICE_DIM)
        self.face_id_map: list[str] = []
        self.voice_id_map: list[str] = []

    def add_face(self, person_id: str, embedding: np.ndarray):
        embedding = embedding / np.linalg.norm(embedding)
        self.face_index.add(embedding.reshape(1, -1))
        self.face_id_map.append(person_id)

    def search_face(self, embedding: np.ndarray, top_k: int = 1) -> tuple[str | None, float]:
        embedding = embedding / np.linalg.norm(embedding)
        scores, indices = self.face_index.search(embedding.reshape(1, -1),top_k)
        score = float(scores[0][0])
        if score < self.THRESHOLD_FACE:
            return None, score
        return self.face_id_map[indices[0][0]], score
    
    def add_voice(self, person_id: str, embedding: np.ndarray):
        embedding = embedding / np.linalg.norm(embedding)
        self.voice_index.add(embedding.reshape(1, -1))
        self.voice_id_map.append(person_id)

    def search_voice(self, embedding: np.ndarray, top_k: int = 1) -> tuple[str | None, float]:
        if self.voice_index.ntotal == 0:
            return None, 0.0
        embedding = embedding / np.linalg.norm(embedding)
        scores, indices = self.voice_index.search(embedding.reshape(1, -1), top_k)
        score = float(scores[0][0])
        if score < self.THRESHOLD_VOICE:
            return None, score
        return self.voice_id_map[indices[0][0]], score

    def save(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.face_index, str(self.data_dir / "face.index"))
        faiss.write_index(self.voice_index, str(self.data_dir / "voice.index"))
        with open(self.data_dir / "id_maps.pkl", "wb") as f:
            pickle.dump({
                "face": self.face_id_map,
                "voice": self.voice_id_map
            }, f)

    def load(self):
        face_path = self.data_dir / "face.index"
        if face_path.exists():
            self.face_index = faiss.read_index(str(face_path))
        voice_path = self.data_dir / "voice.index"
        if voice_path.exists():
            self.voice_index = faiss.read_index(str(voice_path))
        maps_path = self.data_dir / "id_maps.pkl"
        if maps_path.exists():
            with open(maps_path, "rb") as f:
                maps = pickle.load(f)
                self.face_id_map = maps.get("face", [])
                self.voice_id_map = maps.get("voice", [])
