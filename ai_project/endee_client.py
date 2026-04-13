import requests
import numpy as np

class EndeeClient:
    def __init__(self, host="http://localhost:8080"):
        self.host = host
        self.headers = {"Content-Type": "application/json"}
        self._local_db = []

    def _is_server_up(self):
        try:
            requests.get(f"{self.host}/api/v1/health", timeout=0.2)
            return True
        except requests.exceptions.RequestException:
            return False

    def create_index(self, index_name, dimension):
        if self._is_server_up():
            data = {
                "name": index_name,
                "dimension": dimension,
                "metric": "cosine"
            }
            res = requests.post(f"{self.host}/api/v1/index/create", headers=self.headers, json=data, timeout=2)
            return res.status_code in [200, 201]
        return True

    def insert_vectors(self, index_name, vectors, metadata):
        if self._is_server_up():
            data = {
                "index_name": index_name,
                "vectors": vectors,
                "metadata": metadata
            }
            res = requests.post(f"{self.host}/api/v1/index/insert", headers=self.headers, json=data, timeout=2)
            return res.status_code in [200, 201]
            
        for vec, meta in zip(vectors, metadata):
            self._local_db.append({"vector": vec, "metadata": meta})
        return True

    def search_vectors(self, index_name, vector, top_k=3):
        if self._is_server_up():
            data = {
                "index_name": index_name,
                "vector": vector,
                "top_k": top_k
            }
            try:
                res = requests.post(f"{self.host}/api/v1/index/search", headers=self.headers, json=data, timeout=2)
                if res.status_code == 200:
                    response_data = res.json()
                    return response_data.get('results', [])
                return []
            except requests.exceptions.RequestException:
                pass
                
        if not self._local_db:
            return []
            
        def cosine_sim(a, b):
            a, b = np.array(a), np.array(b)
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            return np.dot(a, b) / norm if norm != 0 else 0.0
            
        scored = []
        for item in self._local_db:
            score = cosine_sim(vector, item["vector"])
            scored.append({"score": float(score), "metadata": item["metadata"]})
            
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

