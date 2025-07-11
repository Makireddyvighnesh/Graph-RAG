# Graph-RAG: Basic Graph-Based Implementation of Retrieval-Augmented Generation

This project is a basic implementation of a Graph-based Retrieval-Augmented Generation (Graph-RAG) system. The system constructs a graph-based knowledge representation from a set of documents, retrieves relevant information via graph traversal, and generates answers to questions using a pre-trained generative model.

## System Overview

The system is composed of the following components:

1.  **Text Corpus and Question Creation**: A small corpus of 5 documents on the topic of Generative AI was created, along with 5 corresponding questions and answers.
2.  **Graph Construction and Preprocessing**: The documents are preprocessed by cleaning and normalizing the text. A graph is then constructed where nodes represent sentences and edges represent co-occurrence within the same document.
3.  **Graph-Based Retrieval**: A graph-based retrieval mechanism is implemented to identify the most relevant context for a given question. This is done by converting the question into a vector representation and finding the most similar nodes in the graph.
4.  **Answer Generation**: A pre-trained sequence-to-sequence model (BART) is used to generate answers based on the retrieved context.
5.  **Evaluation**: The generated answers are evaluated against ground truth answers using Exact Match (EM) and F1 Score metrics.

    ### Results

    The system achieved the following results on the test questions:

    *   **Average Exact Match (EM): 0.0000**
    *   **Average F1 Score: 0.51**

## GitHub Repository

This project is not yet in a GitHub repository. To run the project, you can follow these steps:

1.  **Install the required libraries**:

    ```
    pip install -r requirements.txt
    ```

2.  **Run the graph construction script**:

    ```
    python graph_construction.py
    ```

3.  **Run the script**:

    ```
    python main.py
    ```
