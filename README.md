# ResChat: AI-Powered Research Paper Assistant

[![Made with LightRAG](https://img.shields.io/badge/Made_with-LightRAG-blue.svg)](https://github.com/lightrag)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

ResChat is an innovative research paper assistant that combines the power of Knowledge Graphs, Experience-Based Learning, and Dynamic Query Enhancement to revolutionize how researchers interact with academic literature.

## 🌟 Key Features

### 🧠 Intelligent Query Enhancement

- **Dynamic Query Expansion**: Automatically enhances search queries using Groq's Mixtral-8x7b model
- **Academic Term Integration**: Adds relevant terminology and expands abbreviations
- **Methodology Awareness**: Incorporates related research methodologies in search

### 📊 Experience-Adaptive Interface

- **Four Learning Levels**: Customized responses for Beginner, Intermediate, Advanced, and Expert levels
- **Dynamic Content Adaptation**: Adjusts technical depth based on user expertise
- **Contextual Prompting**: Tailors explanations to match user's background

### 🕸️ Visual Knowledge Mapping

- **Interactive Knowledge Graphs**: Visualizes paper relationships and concept connections
- **Dynamic Graph Physics**: Real-time interaction with knowledge structures
- **Entity-Relation Visualization**: Shows connections between research concepts

### 🔍 Dual Search Modes

- **Quick Title Search**: Fast matching using MiniLM embeddings
- **Deep Abstract Analysis**: Comprehensive semantic search through paper abstracts
- **Hybrid Retrieval**: Combines embedding similarity with contextual relevance

## 🛠️ Technical Architecture

```plaintext
reschat/
├── 🧮 Models/
│   ├── embeddings/    # MiniLM-L6-v2 Title Embeddings
│   └── abstracts/     # Research Paper Embeddings
├── 🔄 LightRAG/
│   └── knowledge_graphs/  # Paper-specific Knowledge Graphs
└── 🎯 src/
    ├── app.py            # Streamlit Interface
    ├── rag_manager.py    # LightRAG Integration
    └── recommender.py    # Enhanced Search System
```

## 💫 Unique Capabilities

### Experience-Based Response Generation

```python
prompt_templates = {
    "Beginner": "Explain in simple terms, avoiding technical jargon",
    "Intermediate": "Provide a balanced explanation with some technical details",
    "Advanced": "Give a detailed technical explanation",
    "Expert": "Provide an in-depth analysis with theoretical foundations"
}
```

### Query Enhancement System

```python
# Example of enhanced query generation
Input: "quantum computing"
Enhanced: "quantum computing, qubits, quantum gates, 
          quantum entanglement, quantum algorithms, 
          quantum error correction"
```

## 🚀 Getting Started

1. **Environment Setup**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Keys**

   ```bash
   # .env file
   GROQ_API_KEY=your_groq_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

3. **Launch Application**

   ```bash
   streamlit run src/app.py
   ```

   streamlit run src/app.py
   ```

## 🔋 Core Dependencies

- **LightRAG**: Powers knowledge graph generation and contextual retrieval
- **Groq**: Drives query enhancement using Mixtral-8x7b
- **SentenceTransformers**: Enables semantic search capabilities
- **PyVis**: Powers interactive knowledge graph visualization
- **Streamlit**: Provides the responsive web interface

## 📈 Performance Features

- **Asynchronous Processing**: Non-blocking PDF content extraction
- **Dynamic Graph Physics**: Interactive knowledge visualization
- **Contextual Memory**: Paper-specific knowledge graph generation

## 🎯 Use Cases

1. **Literature Review**: Quickly understand research landscapes
2. **Concept Learning**: Experience-based explanation of complex topics
3. **Deep Paper Analysis**: Detailed Q&A about specific research
