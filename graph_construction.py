import os
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_text(text):
    return text.strip().lower()

def build_graph(doc_path='documents', similarity_threshold=0.75):
    G = nx.Graph()
    sentences = []
    sent_source = []

    for fname in os.listdir(doc_path):
        with open(os.path.join(doc_path, fname), 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.read().split('.') if line.strip()]
            for line in lines:
                cleaned = clean_text(line)
                sentences.append(cleaned)
                sent_source.append(fname)

    embeddings = model.encode(sentences)

    for i, sent in enumerate(sentences):
        vec_str = ",".join(map(str, embeddings[i]))
        G.add_node(sent, vector=vec_str, source=sent_source[i])

    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
            if sim >= similarity_threshold:
                G.add_edge(sentences[i], sentences[j], weight=sim)
    
    return G

if __name__ == '__main__':
    graph = build_graph()
    nx.write_gexf(graph, "knowledge_graph.gexf")
    print("Graph saved to knowledge_graph.gexf")