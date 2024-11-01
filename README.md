# ResChat

ResChat is a personalized research paper recommendation system designed to assist researchers in discovering relevant papers based on their interests. It utilizes advanced retrieval-augmented generation techniques(LightRAG), PyMuPDF for processing PDF content, and Streamlit for an interactive user interface.

---

## 📜 Features

- **Personalized Recommendations**: Finds research papers based on users’ specific interests and input.
- **Full-Content Processing**: Parses and processes entire research papers for precise recommendations and Q&A.
- **Q&A and Summarization**: Provides Q&A capabilities and generates summaries for research papers.
- **Interactive Interface**: A user-friendly Streamlit interface for easy interaction.

---

## 🛠️ Project Structure

The following describes the main components of the project:

```plaintext
reschat/
├── data/
│   └── papers/                  # Research papers in PDF format
├── dataset/                     # Dataset storage
├── lightrag_data/               # LightRAG model data
├── models/
│   ├── abstracts/               # Abstract embeddings (.pkl)
│   ├── embeddings/              # Title embeddings (.pkl)
│   └── sentences/               # Titles (.pkl)
├── notebooks/
│   ├── data-generation.ipynb    # Notebook for data generation
│   └── embeddings-creation.ipynb# Notebook for creating embeddings
├── src/
│   ├── lightrag_data/           # LightRAG configurations
│   ├── app.py                   # Streamlit app entry point
│   ├── rag_manager.py           # Manages RAG and model interactions
│   ├── recommender.py           # Recommendation system implementation
├── .env                         # Environment variables file
├── README.md                    # Project documentation (this file)
├── requirements.txt             # Python dependencies
└── .gitignore                   # Files to ignore in version control
```

---

## 🚀 Quickstart Guide

### Prerequisites

1. **Python**: Install Python 3.7 or higher.
2. **Dependencies**: Install required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**: Add environment variables to the `.env` file for storing sensitive information.

### Running the Application

1. **Start the Streamlit App**:

   ```bash
   streamlit run src/app.py
   ```

2. Open your browser and go to `http://localhost:8501` to interact with ResChat.

---

## 📂 Project Components

### `src/app.py`

- Main entry point for running the ResChat Streamlit app.
- Provides the user interface and handles user interactions.

### `src/rag_manager.py`

- Manages all RAG functionalities, including processing paper content and storing embeddings.
- Uses LightRAG for retrieval-augmented generation to improve recommendation quality.

### `src/recommender.py`

- Implements the recommendation engine for research papers.
- Integrates with LightRAG to deliver personalized recommendations.

---

## 📄 Notebooks

- **data-generation.ipynb**: Generates initial data for the RAG model.
- **embeddings-creation.ipynb**: Creates embeddings from research paper content.

---

## ⚙️ Dependencies

Major dependencies:

- **Streamlit**: Provides the web interface.
- **PyMuPDF (fitz)**: Parses PDFs to extract text.
- **LightRAG**: Powers the retrieval-augmented generation model.

For a complete list, see `requirements.txt`.

---
