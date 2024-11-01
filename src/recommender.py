#recommender.py
import pickle
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class EnhancedPaperRecommender:
    def __init__(self, 
                 embeddings_path='../models/embeddings/embeddings.pkl', 
                 sentences_path='../models/sentences/sentences.pkl',
                 abstracts_path='../models/abstracts/abstracts.pkl',
                 data_path='../dataset/filtered_data.csv'):
        """Initialize with same parameters as before"""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.load_data(embeddings_path, sentences_path, abstracts_path, data_path)
        self.groq_model = ChatGroq(model="llama3-70b-8192")

    def load_data(self, embeddings_path, sentences_path, abstracts_path, data_path):
        """Load data from the specified paths"""
        try:
            with open(embeddings_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            
            with open(sentences_path, 'rb') as f:
                self.sentences = pickle.load(f)
            
            with open(abstracts_path, 'rb') as f:
                self.abstract_embeddings = pickle.load(f)
            
            self.papers_df = pd.read_csv(data_path)
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    async def enhance_query(self, query: str) -> str:
        """Enhance the query using Groq for better search"""
        enhancement_prompt = f"""You are an expert academic research assistant. Your task is to enhance the given research query for semantic search in academic papers.

                Context: The enhanced query will be used for finding relevant academic papers using semantic similarity. The goal is to expand the original query while maintaining focus and relevance.

                Guidelines for enhancement:
                1. Core Terminology:
                - Include fundamental concepts and standard terminology
                - Add widely-accepted alternative terms
                - Include relevant technical jargon

                2. Methodology & Techniques:
                - Add common research methodologies in the field
                - Include relevant analytical techniques
                - Add standard measurement or evaluation methods

                3. Related Concepts:
                - Include closely related theoretical frameworks
                - Add interdisciplinary connections
                - Include relevant sub-fields or specializations

                4. Technical Elements:
                - Expand all relevant abbreviations
                - Include standard nomenclature
                - Add technical specifications where applicable

                Format your response in this exact structure:
                Original Query: {query}

                ENHANCED_QUERY: [primary terms], [methodological terms], [related concepts], [technical specifications]

                Examples:

                Input: "machine learning for healthcare"
                ENHANCED_QUERY: machine learning, artificial intelligence, healthcare analytics, supervised learning, deep learning, neural networks, clinical decision support systems, precision medicine, predictive diagnostics, ML algorithms, biomedical data mining, electronic health records (EHR)

                Input: "quantum computing"
                ENHANCED_QUERY: quantum computing, quantum information processing, quantum mechanics, quantum gates, quantum circuits, quantum algorithms, quantum entanglement, quantum superposition, quantum coherence, qubits, quantum error correction, quantum supremacy

                Now, enhance the following query while maintaining high relevance and avoiding topic drift:
                {query}"""
        
        response =  self.groq_model.invoke([{"role": "user", "content": enhancement_prompt}])
        enhanced_query = response.content.split("Enhanced Query:")[-1].strip()
        return enhanced_query

    async def recommend_by_abstracts(self, 
                                    query: str, 
                                    top_k: int = 5,
                                    experience_level: str = "Intermediate") -> List[Dict[str, Any]]:
        enhanced_query = await self.enhance_query(query)
        enhanced_embedding = self.model.encode(enhanced_query)
        combined_scores = util.cos_sim(self.abstract_embeddings, enhanced_embedding)
        top_indices = torch.topk(combined_scores, dim=0, k=5, sorted=True).indices
        
        recommendations = []
        for idx in top_indices:
            title = self.sentences[idx.item()]
            paper_data = self.papers_df[self.papers_df['titles'] == title].iloc[0]
            
            recommendations.append({
                'title': title,
                'abstract': paper_data['abstracts'],
                'url': paper_data['urls'],
                'relevance_score': combined_scores[idx].item(),
                'enhanced_query': enhanced_query
            })
        
        return recommendations
    
    async def recommend_by_titles(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Async version of title-based recommendation"""
        query_embedding = self.model.encode(query)
        cosine_scores = util.cos_sim(self.embeddings, query_embedding)
        top_indices = torch.topk(cosine_scores, dim=0, k=top_k, sorted=True).indices
        
        recommendations = []
        for idx in top_indices:
            title = self.sentences[idx.item()]
            paper_data = self.papers_df[self.papers_df['titles'] == title].iloc[0]
            
            recommendations.append({
                'title': title,
                'abstract': paper_data['abstracts'],
                'url': paper_data['urls'],
                'relevance_score': cosine_scores[idx].item()
            })
        
        return recommendations