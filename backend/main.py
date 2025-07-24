from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import openai
import os
import faiss
import numpy as np
from dotenv import load_dotenv

# Load secrets
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# --- Simulated Clauses (normally you'd parse documents) ---
CLAUSES = [
    "Knee surgeries are covered after a 6-month waiting period.",
    "Surgeries must be done in Tier-1 cities including Pune, Mumbai, and Delhi.",
    "Policies less than 6 months old are not eligible for major surgeries.",
    "Pre-existing conditions are excluded for the first 12 months.",
    "The maximum payout for surgeries is ₹2,00,000 if eligible."
]

from openai import OpenAI

client = OpenAI()

def embed_texts(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [np.array(e.embedding) for e in response.data]

#Embedding Clauses once
CLAUSE_EMBEDDINGS =embed_texts(CLAUSES)
DIM = len(CLAUSE_EMBEDDINGS[0])
index = faiss.IndexFlatL2(DIM)#Uses Euclidean distance to compare vectors
index.add(np.array(CLAUSE_EMBEDDINGS))#converts all the vector into numpy array and adds them to Faiss index

class QueryInput(BaseModel):
    query: str

class Justification(BaseModel):
    clause:str
    reason:str

class DecisionResponse(BaseModel):
    decision: str
    amount: Optional[str] =None
    justification: List[Justification]

# semantic Retrieval

def get_clauses(query, k=3):
    q_embedding = embed_texts([query])[0].reshape(1,-1)
    D,I = index.search(q_embedding,k)
    return [CLAUSES[i] for i in [0]]

# --- Decision Engine (LLM Agent) ---
def generate_decision(query: str, relevant_clauses: List[str]) -> DecisionResponse:
    prompt = f"""
You are an insurance decision engine. Based on the following query and policy clauses, decide whether the procedure is approved.

Query:
"{query}"

Relevant Clauses:
{chr(10).join([f"- {c}" for c in relevant_clauses])}

Respond in JSON format like this:
{{
  "decision": "approved/rejected",
  "amount": "number in INR or null",
  "justification": [
    {{"clause": "...", "reason": "..."}}
  ]
}}
"""

    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert insurance decision assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    import json
    return DecisionResponse(**json.loads(completion.choices[0].message["content"]))

# --- API Endpoint ---
@app.post("/api/decide", response_model=DecisionResponse)
def decide_route(data: QueryInput):
    try:
        clauses = get_clauses(data.query)
        result = generate_decision(data.query, clauses)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))