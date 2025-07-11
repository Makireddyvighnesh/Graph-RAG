import json
import networkx as nx
import numpy as np
from graph_retrieval import get_relevant_text_from_graph
from answer_generation import generate_answer
from evaluation import exact_match, f1_score

def main():
    G = nx.read_gexf("knowledge_graph.gexf")
    for node in G.nodes():
        G.nodes[node]['vector'] = G.nodes[node]['vector']

    with open("questions_and_answers.json") as f:
        qa_pairs = json.load(f)

    em_scores, f1_scores = [], []

    for i, qa in enumerate(qa_pairs):
        question = qa['question']
        reference = qa['answer']
        context = get_relevant_text_from_graph(G, question)
        
        full_context = " ".join(context)
        prediction = generate_answer(question, full_context)

        em = exact_match(prediction, reference)
        f1 = f1_score(prediction, reference)

        print(f"Q{i+1}: {question}")
        print("Context retrieved:", context)
        print(f"Prediction: {prediction}")
        print(f"Reference: {reference}")
        print(f"EM: {em}, F1: {f1:.2f}\\n")

        em_scores.append(em)
        f1_scores.append(f1)

    print(f"Avg EM: {np.mean(em_scores):.2f}, Avg F1: {np.mean(f1_scores):.2f}")

if __name__ == "__main__":
    main()