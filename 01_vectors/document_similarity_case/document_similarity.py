"""
Document Similarity with Dot Product
===================================

This module demonstrates how to use dot product for document similarity
in Natural Language Processing and Information Retrieval.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

def simple_document_similarity():
    """Demonstrate basic document similarity using dot product."""
    print("=== Simple Document Similarity ===")
    
    # Sample documents
    doc1 = "machine learning artificial intelligence"
    doc2 = "artificial intelligence deep learning"
    doc3 = "cooking recipes food preparation"
    
    print(f"Document 1: '{doc1}'")
    print(f"Document 2: '{doc2}'")
    print(f"Document 3: '{doc3}'")
    
    # Create vocabulary (unique words)
    all_words = set(doc1.split() + doc2.split() + doc3.split())
    vocabulary = sorted(list(all_words))
    print(f"\nVocabulary: {vocabulary}")
    
    # Convert documents to vectors (word frequencies)
    def doc_to_vector(doc, vocab):
        words = doc.split()
        word_count = Counter(words)
        vector = [word_count.get(word, 0) for word in vocab]
        return np.array(vector)
    
    # Create document vectors
    v1 = doc_to_vector(doc1, vocabulary)
    v2 = doc_to_vector(doc2, vocabulary)
    v3 = doc_to_vector(doc3, vocabulary)
    
    print(f"\nDocument vectors:")
    print(f"Doc1 vector: {v1}")
    print(f"Doc2 vector: {v2}")
    print(f"Doc3 vector: {v3}")
    
    # Calculate similarities using dot product
    sim_1_2 = np.dot(v1, v2)
    sim_1_3 = np.dot(v1, v3)
    sim_2_3 = np.dot(v2, v3)
    
    print(f"\nSimilarity scores (dot product):")
    print(f"Doc1 · Doc2 = {sim_1_2}")
    print(f"Doc1 · Doc3 = {sim_1_3}")
    print(f"Doc2 · Doc3 = {sim_2_3}")
    
    # Interpret results
    print(f"\nInterpretation:")
    print(f"- Doc1 and Doc2 are most similar (score: {sim_1_2})")
    print(f"- Doc3 is least similar to others (scores: {sim_1_3}, {sim_2_3})")
    print()

def document_similarity_with_charts():
    """Visualize document similarity with charts."""
    print("=== Document Similarity Visualization ===")
    
    # Sample documents about different topics
    documents = [
        "machine learning artificial intelligence neural networks",
        "artificial intelligence deep learning computer vision",
        "cooking recipes food preparation kitchen",
        "machine learning data science algorithms",
        "food recipes cooking techniques ingredients"
    ]
    
    doc_names = ["ML/AI", "AI/DL", "Cooking", "ML/DS", "Food"]
    
    print("Documents:")
    for i, doc in enumerate(documents):
        print(f"{i+1}. {doc_names[i]}: '{doc}'")
    
    # Create vocabulary
    all_words = set()
    for doc in documents:
        all_words.update(doc.split())
    vocabulary = sorted(list(all_words))
    
    # Convert documents to vectors
    def doc_to_vector(doc, vocab):
        words = doc.split()
        word_count = Counter(words)
        vector = [word_count.get(word, 0) for word in vocab]
        return np.array(vector)
    
    # Create document vectors
    doc_vectors = [doc_to_vector(doc, vocabulary) for doc in documents]
    
    # Calculate similarity matrix
    n_docs = len(documents)
    similarity_matrix = np.zeros((n_docs, n_docs))
    
    for i in range(n_docs):
        for j in range(n_docs):
            similarity_matrix[i, j] = np.dot(doc_vectors[i], doc_vectors[j])
    
    print(f"\nSimilarity Matrix (dot product):")
    print("     ", end="")
    for name in doc_names:
        print(f"{name:>8}", end="")
    print()
    
    for i, name in enumerate(doc_names):
        print(f"{name:>5}", end="")
        for j in range(n_docs):
            print(f"{similarity_matrix[i, j]:>8.1f}", end="")
        print()
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Similarity matrix heatmap
    im = ax1.imshow(similarity_matrix, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(n_docs))
    ax1.set_yticks(range(n_docs))
    ax1.set_xticklabels(doc_names, rotation=45)
    ax1.set_yticklabels(doc_names)
    ax1.set_title('Document Similarity Matrix')
    
    # Add text annotations
    for i in range(n_docs):
        for j in range(n_docs):
            text = ax1.text(j, i, f'{similarity_matrix[i, j]:.1f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbar
    plt.colorbar(im, ax=ax1, label='Similarity Score')
    
    # Plot 2: Bar chart of similarities
    # Find most similar pairs (excluding self-similarity)
    similarities = []
    pairs = []
    
    for i in range(n_docs):
        for j in range(i+1, n_docs):
            similarities.append(similarity_matrix[i, j])
            pairs.append(f"{doc_names[i]}-{doc_names[j]}")
    
    # Sort by similarity
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_similarities = [similarities[i] for i in sorted_indices]
    sorted_pairs = [pairs[i] for i in sorted_indices]
    
    bars = ax2.bar(range(len(sorted_pairs)), sorted_similarities, 
                   color='skyblue', alpha=0.7)
    ax2.set_xlabel('Document Pairs')
    ax2.set_ylabel('Similarity Score')
    ax2.set_title('Document Pair Similarities')
    ax2.set_xticks(range(len(sorted_pairs)))
    ax2.set_xticklabels(sorted_pairs, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, sorted_similarities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("Visualization shows:")
    print("- Heatmap of similarity matrix")
    print("- Bar chart of document pair similarities")
    print("- Higher values indicate more similar documents")
    print()

def search_engine_simulation():
    """Simulate a simple search engine using dot product."""
    print("=== Search Engine Simulation ===")
    
    # Document collection
    documents = [
        "machine learning artificial intelligence neural networks deep learning",
        "cooking recipes food preparation kitchen techniques ingredients",
        "artificial intelligence computer vision image recognition",
        "data science machine learning algorithms statistics",
        "food recipes cooking methods kitchen tools",
        "neural networks deep learning artificial intelligence"
    ]
    
    doc_titles = [
        "ML/AI Overview",
        "Cooking Basics", 
        "Computer Vision",
        "Data Science",
        "Food Preparation",
        "Deep Learning"
    ]
    
    # User query
    query = "machine learning artificial intelligence"
    print(f"Query: '{query}'")
    print(f"\nDocument Collection:")
    for i, (title, doc) in enumerate(zip(doc_titles, documents)):
        print(f"{i+1}. {title}: '{doc}'")
    
    # Create vocabulary
    all_words = set()
    for doc in documents:
        all_words.update(doc.split())
    vocabulary = sorted(list(all_words))
    
    # Convert documents and query to vectors
    def doc_to_vector(doc, vocab):
        words = doc.split()
        word_count = Counter(words)
        vector = [word_count.get(word, 0) for word in vocab]
        return np.array(vector)
    
    # Create vectors
    doc_vectors = [doc_to_vector(doc, vocabulary) for doc in documents]
    query_vector = doc_to_vector(query, vocabulary)
    
    print(f"\nQuery vector: {query_vector}")
    print(f"Vocabulary: {vocabulary}")
    
    # Calculate relevance scores using dot product
    relevance_scores = []
    for i, doc_vector in enumerate(doc_vectors):
        score = np.dot(query_vector, doc_vector)
        relevance_scores.append((score, i, doc_titles[i], documents[i]))
    
    # Sort by relevance (highest first)
    relevance_scores.sort(reverse=True)
    
    print(f"\nSearch Results (ranked by relevance):")
    print("Rank | Score | Title | Document")
    print("-" * 60)
    
    for rank, (score, idx, title, doc) in enumerate(relevance_scores, 1):
        print(f"{rank:4d} | {score:5.1f} | {title:15s} | {doc[:30]}...")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Extract data for plotting
    scores = [score for score, _, _, _ in relevance_scores]
    titles = [title for _, _, title, _ in relevance_scores]
    
    # Create bar chart
    bars = plt.bar(range(len(scores)), scores, color='lightcoral', alpha=0.7)
    plt.xlabel('Document Rank')
    plt.ylabel('Relevance Score')
    plt.title(f'Search Results for Query: "{query}"')
    plt.xticks(range(len(titles)), titles, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nTop result: '{relevance_scores[0][2]}' with score {relevance_scores[0][0]}")
    print()

def document_clustering():
    """Group similar documents using dot product similarity."""
    print("=== Document Clustering ===")
    
    # Sample documents
    documents = [
        "machine learning artificial intelligence neural networks",
        "cooking recipes food preparation kitchen",
        "artificial intelligence deep learning computer vision", 
        "food recipes cooking techniques ingredients",
        "data science machine learning algorithms",
        "kitchen tools cooking equipment food preparation"
    ]
    
    doc_names = ["ML1", "Cooking1", "AI1", "Food1", "DS1", "Kitchen1"]
    
    print("Documents to cluster:")
    for name, doc in zip(doc_names, documents):
        print(f"{name}: '{doc}'")
    
    # Create vocabulary and vectors
    all_words = set()
    for doc in documents:
        all_words.update(doc.split())
    vocabulary = sorted(list(all_words))
    
    def doc_to_vector(doc, vocab):
        words = doc.split()
        word_count = Counter(words)
        vector = [word_count.get(word, 0) for word in vocab]
        return np.array(vector)
    
    doc_vectors = [doc_to_vector(doc, vocabulary) for doc in documents]
    
    # Calculate similarity matrix
    n_docs = len(documents)
    similarity_matrix = np.zeros((n_docs, n_docs))
    
    for i in range(n_docs):
        for j in range(n_docs):
            similarity_matrix[i, j] = np.dot(doc_vectors[i], doc_vectors[j])
    
    # Simple clustering: group documents with similarity > threshold
    threshold = 3.0  # Minimum similarity for clustering
    clusters = []
    used = set()
    
    for i in range(n_docs):
        if i in used:
            continue
        
        cluster = [i]
        used.add(i)
        
        for j in range(i+1, n_docs):
            if j not in used and similarity_matrix[i, j] >= threshold:
                cluster.append(j)
                used.add(j)
        
        clusters.append(cluster)
    
    print(f"\nClustering Results (threshold = {threshold}):")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1}: {[doc_names[idx] for idx in cluster]}")
        for idx in cluster:
            print(f"  - {doc_names[idx]}: '{documents[idx]}'")
    
    # Visualize clusters
    plt.figure(figsize=(10, 6))
    
    # Create a simple 2D representation using PCA-like approach
    # For simplicity, we'll use the first two dimensions of the similarity matrix
    x_coords = similarity_matrix[:, 0]  # Similarity to first document
    y_coords = similarity_matrix[:, 1]  # Similarity to second document
    
    # Plot documents
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        plt.scatter(x, y, c=colors[i], s=100, alpha=0.7, label=doc_names[i])
        plt.annotate(doc_names[i], (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Similarity to ML1')
    plt.ylabel('Similarity to Cooking1')
    plt.title('Document Clustering Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("Visualization shows documents in 2D space based on similarity")
    print()

def tf_idf_similarity():
    """Demonstrate TF-IDF weighted document similarity."""
    print("=== TF-IDF Document Similarity ===")
    
    # Sample documents
    documents = [
        "machine learning artificial intelligence",
        "artificial intelligence deep learning", 
        "cooking recipes food preparation",
        "machine learning data science algorithms"
    ]
    
    doc_names = ["ML/AI", "AI/DL", "Cooking", "ML/DS"]
    
    print("Documents:")
    for name, doc in zip(doc_names, documents):
        print(f"{name}: '{doc}'")
    
    # Create vocabulary
    all_words = set()
    for doc in documents:
        all_words.update(doc.split())
    vocabulary = sorted(list(all_words))
    
    # Calculate TF (Term Frequency)
    def calculate_tf(doc, vocab):
        words = doc.split()
        word_count = Counter(words)
        total_words = len(words)
        tf = [word_count.get(word, 0) / total_words for word in vocab]
        return np.array(tf)
    
    # Calculate IDF (Inverse Document Frequency)
    def calculate_idf(docs, vocab):
        n_docs = len(docs)
        idf = []
        for word in vocab:
            docs_with_word = sum(1 for doc in docs if word in doc.split())
            idf_value = np.log(n_docs / docs_with_word) if docs_with_word > 0 else 0
            idf.append(idf_value)
        return np.array(idf)
    
    # Calculate TF-IDF vectors
    tf_vectors = [calculate_tf(doc, vocabulary) for doc in documents]
    idf_vector = calculate_idf(documents, vocabulary)
    
    tf_idf_vectors = [tf * idf_vector for tf in tf_vectors]
    
    print(f"\nVocabulary: {vocabulary}")
    print(f"IDF values: {idf_vector}")
    
    # Calculate similarities using dot product
    print(f"\nTF-IDF Similarity Matrix:")
    n_docs = len(documents)
    similarity_matrix = np.zeros((n_docs, n_docs))
    
    for i in range(n_docs):
        for j in range(n_docs):
            similarity_matrix[i, j] = np.dot(tf_idf_vectors[i], tf_idf_vectors[j])
    
    # Print similarity matrix
    print("     ", end="")
    for name in doc_names:
        print(f"{name:>8}", end="")
    print()
    
    for i, name in enumerate(doc_names):
        print(f"{name:>5}", end="")
        for j in range(n_docs):
            print(f"{similarity_matrix[i, j]:>8.3f}", end="")
        print()
    
    print("\nTF-IDF gives more weight to rare, important words")
    print("Higher values indicate more similar documents")
    print()

if __name__ == "__main__":
    print("Document Similarity with Dot Product\n")
    simple_document_similarity()
    document_similarity_with_charts()
    search_engine_simulation()
    document_clustering()
    tf_idf_similarity()
