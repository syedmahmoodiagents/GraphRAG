import os
from dotenv import load_dotenv
load_dotenv()
os.getenv("HF_TOKEN")

from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
     

doc = {
    "Transformer": """
The Transformer is a neural network architecture introduced in 2017.
It relies entirely on attention mechanisms.
""",
    "Attention": """
Attention allows the model to focus on relevant parts of the input sequence.
Self-attention computes relationships within the same sequence.
""",
    "Encoder": """
The encoder maps input tokens into contextual embeddings.
""",
    "Decoder": """
The decoder generates output tokens one step at a time.
"""
}


graph_db = {}

def ingest_document(doc: dict):
    for section, text in doc.items():
        graph_db[section] = {
            "content": text.strip(),
            "neighbors": []
        }

    # add relationships (could be LLM-driven)
    graph_db["Transformer"]["neighbors"] = ["Attention", "Encoder", "Decoder"]
    graph_db["Attention"]["neighbors"] = ["Self-Attention"]

ingest_document(doc)

class GraphRAGState(TypedDict):
    query: str
    entities: List[str]
    matched_nodes: List[Dict]
    graph_context: str
    answer: str


def extract_entities(state: GraphRAGState):
    query = state["query"]
    entities = [
        node for node in graph_db.keys()
        if node.lower() in query.lower()
    ]

    return {"entities": entities}

def traverse_graph(state: GraphRAGState):
    matched = []

    for entity in state["entities"]:
        node = graph_db.get(entity)
        if not node:
            continue

        matched.append({
            "entity": entity,
            "content": node["content"],
            "neighbors": node["neighbors"]
        })

        for neighbor in node["neighbors"]:
            if neighbor in graph_db:
                matched.append({
                    "entity": neighbor,
                    "content": graph_db[neighbor]["content"],
                    "neighbors": graph_db[neighbor]["neighbors"]
                })

    return {"matched_nodes": matched}



def build_context(state: GraphRAGState):
    context = ""
    for node in state["matched_nodes"]:
        context += f"""
            Entity: {node['entity']}
            Description: {node['content']}
            Related: {', '.join(node['neighbors'])}
        """
    return {"graph_context": context}


llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b"))

def generate_answer(state: GraphRAGState):
    prompt = f"""
        You must answer ONLY using the context below.
        If the answer is not in the context, say "I don't know".

        Context:
        {state['graph_context']}

        Question:
        {state['query']}
        """
    response = llm.invoke(prompt)
    return {"answer": response.content}


builder = StateGraph(GraphRAGState)

builder.add_node("extract", extract_entities)
builder.add_node("traverse", traverse_graph)
builder.add_node("context", build_context)
builder.add_node("generate", generate_answer)

builder.set_entry_point("extract")
builder.add_edge("extract", "traverse")
builder.add_edge("traverse", "context")
builder.add_edge("context", "generate")
builder.add_edge("generate", END)

graph = builder.compile()

result = graph.invoke({
    "query": "Explain how Transformer attention works"
})

print(result["answer"])