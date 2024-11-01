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
        self.groq_model = ChatGroq(model="mixtral-8x7b-32768")

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
        enhancement_prompt = f"""Given the research interest: "{query}"
        Please enhance this query by:
        1. Adding relevant academic terminology
        2. Expanding abbreviations
        3. Including related concepts
        4. Incorporating key methodologies
        Keep the enhanced query concise but comprehensive.
        Format: Enhanced Query: <query>"""
        
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