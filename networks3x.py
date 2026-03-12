import os
from dotenv import load_dotenv
load_dotenv()
os.getenv("HF_TOKEN")

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import networkx as nx


documents = [
    "LangGraph is used to build agent workflows.",
    "CrewAI is a framework for multi-agent orchestration.",
    "LangChain provides tools for building LLM applications.",
    "GraphRAG retrieves knowledge using graph relationships."
]


def extract_entities(text):
    entities = []
    for word in text.split():
        word = word.replace(".", "")
        if word and word[0].isupper():
            entities.append(word)
    return entities

G = nx.Graph()

chunks = []

for i, doc in enumerate(documents):
    chunk_id = f"chunk_{i}"
    G.add_node(chunk_id, type="chunk", text=doc)
    entities = extract_entities(doc)
    for entity in entities:
        if not G.has_node(entity):
            G.add_node(entity, type="entity")
        G.add_edge(entity, chunk_id)


query = "How does LangGraph help in agents?"
query_entities = extract_entities(query)

retrieved_chunks = []
for entity in query_entities:
    if entity in G:
        neighbors = list(G.neighbors(entity))
        for n in neighbors:
            if G.nodes[n]["type"] == "chunk":
                retrieved_chunks.append(G.nodes[n]["text"])

context = "\n".join(set(retrieved_chunks))

print("\nRetrieved Context:\n")
print(context)


llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b"))

def generate_answer():
    prompt = f"""
        You must answer ONLY using the context below.
        If the answer is not in the context, say "I don't know".

        Context:
        {context}

        Question:
        {query}
        """
    response = llm.invoke(prompt)
    return {"answer": response.content}

result = generate_answer()
print(result["answer"])