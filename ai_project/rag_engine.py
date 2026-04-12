import os
from sentence_transformers import SentenceTransformer
from endee_client import EndeeClient
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class MitigationAgent:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = EndeeClient()
        self.index_name = "verified_db"
        self.dimension = self.model.get_sentence_embedding_dimension()
        try:
            self.client.create_index(self.index_name, self.dimension)
        except Exception as e:
            raise ConnectionError("Make sure the Endee Vector Database is running on port 8080. Original Error: " + str(e))

        self._seed_initial_facts()

    def _seed_initial_facts(self):
        sample_facts = [
            "An apple is a sweet, edible fruit historically known to be produced by an apple tree.",
            "Endee is an ultra-fast, C++ based open-source vector database built for high scalability.",
            "The capital of France is Paris."
        ]
        for fact in sample_facts:
            self.load_document(fact)

    def load_document(self, text):
        chunk_size = 300
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        for c in chunks:
            vector = self.model.encode(c).tolist()
            self.client.insert_vectors(self.index_name, [vector], [{"text": c}])

    def fetch_context(self, query):
        q_vec = self.model.encode(query).tolist()
        res = self.client.search_vectors(self.index_name, q_vec, top_k=1)
        if res:
            return res[0].get("metadata", {}).get("text", "")
        return ""

    def generate_and_verify(self, query):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "Configuration Error: GROQ_API_KEY is not set in environment.", "ERROR", ""
            
        groq_client = Groq(api_key=api_key)
        context = self.fetch_context(query)
        
        if not context:
            prompt = query
            status_flag = "UNKNOWN"
        else:
            prompt = f"Using ONLY the following context, answer the query factually.\n\nContext: {context}\n\nQuery: {query}"
        
        raw_response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.7
        ).choices[0].message.content
        
        if context:
            verification_prompt = f"Fact/Context: {context}\nLLM Response: '{raw_response}'\nDoes the LLM Response contradict the Fact? Reply with exact isolated word: 'HALLUCINATION' if it contradicts, or 'FACTUAL' if it matches."
            verification = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": verification_prompt}],
                model="llama3-8b-8192",
                temperature=0.0
            ).choices[0].message.content
        else:
            verification = "UNVERIFIED"

        return raw_response, verification, context
