# rag_manager.py
import os
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm import openai_complete_if_cache, openai_embedding, gpt_4o_mini_complete
from dotenv import load_dotenv
import networkx as nx
from pyvis.network import Network

load_dotenv()

class RAGManager:
    def __init__(self, working_dir: str):
        self.working_dir = working_dir
        self.setup_directories()
        self.rag = None
        self.graph_path = None
    
    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
    
    @staticmethod
    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
        """Async LLM function using OpenAI API"""
        return await openai_complete_if_cache(
            "llama3-70b-8192",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            **kwargs
        )
    
    @staticmethod
    async def embedding_func(texts: list[str]) -> np.ndarray:
        """Async embedding function using OpenAI API"""
        return await openai_embedding(
            texts,
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
            #api_key=openai_api_key,
            base_url="https://api.openai.com/v1"
        )
    
    def initialize_rag(self) -> LightRAG:
        """Initialize a new LightRAG instance"""
        return LightRAG(
            working_dir=self.working_dir,
            llm_model_func=gpt_4o_mini_complete,
            #llm_model_func=self.llm_model_func,
            # embedding_func=EmbeddingFunc(
            #     embedding_dim=1536,
            #     max_token_size=8192,
            #     func=self.embedding_func
            # )
        )
    
    async def process_paper(self, paper_text: str, paper_id: str) -> None:
        """Process a new paper by creating a new RAG instance and inserting the paper"""
        # Create a new RAG instance for this paper
        self.rag = self.initialize_rag()
        
        # Insert the paper content
        await self.rag.ainsert(paper_text)
        
        # Generate and save the knowledge graph
        self.graph_path = os.path.join(self.working_dir, "knowledge_graph.html")
        self.visualize_knowledge_graph()
    
    def visualize_knowledge_graph(self):
        """Visualize the LightRAG knowledge graph"""
        graph_path = os.path.join(self.working_dir, "graph_chunk_entity_relation.graphml")
        if os.path.exists(graph_path):
            G = nx.read_graphml(graph_path)
            
            net = Network(notebook=True, height="500px", width="100%",
                         bgcolor="#ffffff", font_color="#000000")
            
            net.from_nx(G)
            
            net.toggle_physics(True)
            net.show_buttons(filter_=['physics'])
            
            net.save_graph(self.graph_path)
    
    async def query_papers(self, query: str, context: str, mode: str = "hybrid") -> str:
        """Query the current RAG instance"""
        if not self.rag:
            raise ValueError("No RAG instance available. Please process a paper first.")
        
        query = query + " " + context
        
        result = await self.rag.aquery(
            query,
            param=QueryParam(
                mode=mode,
                response_type="Multiple Paragraphs",
                top_k=3
            )
        )
        
        return result