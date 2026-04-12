import requests

class EndeeClient:
    def __init__(self, host="http://localhost:8080"):
        self.host = host
        self.headers = {"Content-Type": "application/json"}

    def create_index(self, index_name, dimension):
        data = {
            "name": index_name,
            "dimension": dimension,
            "metric": "cosine"
        }
        res = requests.post(f"{self.host}/api/v1/index/create", headers=self.headers, json=data)
        return res.status_code in [200, 201]

    def insert_vectors(self, index_name, vectors, metadata):
        data = {
            "index_name": index_name,
            "vectors": vectors,
            "metadata": metadata
        }
        res = requests.post(f"{self.host}/api/v1/index/insert", headers=self.headers, json=data)
        return res.status_code in [200, 201]

    def search_vectors(self, index_name, vector, top_k=3):
        data = {
            "index_name": index_name,
            "vector": vector,
            "top_k": top_k
        }
        try:
            res = requests.post(f"{self.host}/api/v1/index/search", headers=self.headers, json=data)
            if res.status_code == 200:
                response_data = res.json()
                return response_data.get('results', [])
            return []
        except requests.exceptions.RequestException:
            return []
