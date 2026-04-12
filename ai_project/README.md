# Context-Aware Hallucination Mitigation System

An enterprise-grade application built to aggressively mitigate Large Language Model (LLM) hallucinations by integrating strict contextual grounding with the high-performance **Endee Vector Database**.

## Architecture Overview

1. **Knowledge Ingestion**: Structured documents are chunked, embedded via `sentence-transformers`, and directly indexed into the Endee Vector Database.
2. **Retrieve & Ground**: Upon receiving a user prompt, the system queries the Endee vector index to fetch highly relevant facts.
3. **Generation**: The context and prompt are passed directly to the Groq LLM API.
4. **Factuality Verification**: The model's response is semantically cross-verified against the retrieved context to detect contradictions, ensuring zero hallucination spread.
5. **Graceful UI Degradation**: Engineered with robust state handling, allowing the Streamlit UI dashboard to safely render in a disabled state even if the underlying Vector DB connection is interrupted.

## Core Features
* Sub-millisecond similarity search integration with local Endee implementation.
* Contextual Grounding Engine overriding standard generative responses.
* Zero-dependency containerization out-of-the-box (`docker-compose`).
* Beautiful, responsive UI built natively.

## Requirements

- Python 3.10+
- Docker & Docker Compose
- Groq API Key

## Setup & Execution

### 1. Start the Endee Vector Database
Endee is required to run this application. Connect to the active instance or use Docker Compose to spin up the local server from the Endee repository root.
```bash
docker compose up -d
```

### 2. Install Project Dependencies
```bash
cd ai_project
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Verify your `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```
*(Ensure it contains your active key.)*

### 4. Run the Application locally
```bash
streamlit run app.py
```

### 5. Running via Docker
If you'd like to containerize the UI project:
```bash
docker build -t endee-ai-ui .
docker run -p 8501:8501 --env-file .env endee-ai-ui
```
