import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_relevant_text_from_graph(G, question, top_k=3):
    q_vec = model.encode([question])
    sentence_scores = []

    for node in G.nodes():
        vec = np.fromstring(G.nodes[node]['vector'], sep=',')
        sim = cosine_similarity([q_vec[0]], [vec])[0][0]
        sentence_scores.append((node, sim))

    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    top_context = [sent for sent, _ in sentence_scores[:top_k]]
    return top_context

if __name__ == '__main__':
    G = nx.read_gexf("knowledge_graph.gexf")
    for node in G.nodes():
        G.nodes[node]['vector'] = G.nodes[node]['vector']