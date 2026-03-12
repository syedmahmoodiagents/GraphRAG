import os
from dotenv import load_dotenv
load_dotenv()
os.getenv("HF_TOKEN")

# from typing import TypedDict, List, Dict
# from langgraph.graph import StateGraph, END
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import networkx as nx

documents = [
    "LangGraph is used to build agent workflows.",
    "CrewAI is a multi-agent orchestration framework.",
    "LangChain provides tools for building LLM applications.",
    "GraphRAG retrieves knowledge using graph relationships."
]

chunks = []
for doc in documents:
    sentences = doc.split(".")
    for s in sentences:
        if s.strip():
            chunks.append(s.strip())

def extract_entities(text):
    entities = []
    words = text.split()
    for w in words:
        if w[0].isupper():
            entities.append(w)
    return entities


G = nx.Graph()
for chunk in chunks:
    entities = extract_entities(chunk)
    for entity in entities:
        G.add_node(entity)

    for i in range(len(entities)):
        for j in range(i+1, len(entities)):
            G.add_edge(entities[i], entities[j], text=chunk)

print("Nodes:", G.nodes())
print("Edges:", G.edges(data=True))

query = "How does LangGraph help in agents?"
query_entities = extract_entities(query)