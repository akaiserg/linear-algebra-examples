# Document Similarity with Dot Product

## Problem Statement
Document similarity is a fundamental problem in Natural Language Processing (NLP) and Information Retrieval. We need to:

- **Compare documents** to find similar content
- **Rank search results** by relevance
- **Detect plagiarism** and duplicate content
- **Group similar documents** for organization
- **Recommend related articles** to users

## Dot Product Approach
We use **dot product** to measure document similarity by:

1. **Vector Representation**: Convert documents to numerical vectors
2. **Word Frequency**: Count occurrences of each word
3. **Similarity Calculation**: `similarity = doc1_vector · doc2_vector`
4. **Ranking**: Higher dot product = more similar documents

## Key Concepts Demonstrated
- **Bag of Words**: Represent documents as word frequency vectors
- **Dot Product**: `v1 · v2 = v1[0]*v2[0] + v1[1]*v2[1] + ... + v1[n]*v2[n]`
- **Similarity Scoring**: Higher values indicate more similar documents
- **Vector Space Model**: Documents as points in high-dimensional space
- **TF-IDF**: Term Frequency-Inverse Document Frequency weighting

## Learning Objectives
- Understand how text documents become numerical vectors
- Learn to use dot product for similarity measurement
- Apply document similarity in real-world scenarios
- Visualize similarity relationships with charts
- Practice NLP preprocessing techniques

## Files
- `document_similarity.py`: Main implementation with examples
- `README.md`: This documentation

## Business Applications
- **Search Engines**: Google, Bing ranking algorithms
- **Content Management**: Organize and categorize documents
- **Plagiarism Detection**: Academic and professional use
- **News Aggregation**: Group similar articles
- **Recommendation Systems**: "Related articles" features
- **Legal Research**: Find similar case law
- **Customer Support**: Match queries to knowledge base
