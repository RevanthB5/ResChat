#app.py
import fitz
import requests
import streamlit as st
from rag_manager import RAGManager
import asyncio
from recommender import EnhancedPaperRecommender
import streamlit.components.v1 as components


def init_session_state():
    """Initialize all session state variables"""
    if 'recommender' not in st.session_state:
        st.session_state.recommender = EnhancedPaperRecommender()
    
    if 'selected_paper' not in st.session_state:
        st.session_state.selected_paper = None
    
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = None
    
    if 'paper_answers' not in st.session_state:
        st.session_state.paper_answers = {}
    
    if 'recommended_papers' not in st.session_state:
        st.session_state.recommended_papers = []
    
    if 'current_rag' not in st.session_state:
        st.session_state.current_rag = None


async def fetch_pdf_content(pdf_url):
    """Download and extract text content from a PDF using PyMuPDF."""
    pdf_content = ""
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with fitz.open(stream=response.content, filetype="pdf") as pdf:
            for page in pdf:
                pdf_content += page.get_text()
    else:
        pdf_content = "Could not retrieve PDF content."
    return pdf_content

async def create_paper_card(paper, index):
    """Create a visually appealing card for each paper"""
    with st.container():
        st.markdown(f"""
        <div style='padding: 1rem; border-radius: 0.5rem; background: #f0f2f6; margin-bottom: 1rem;'>
            <h3 style='color: #677997'>{paper['title']}</h3>
            <div style='color: #677997; margin-bottom: 0.5rem;'>
                Relevance Score: {paper.get('relevance_score', 0):.2f}
            </div>
        """, unsafe_allow_html=True)
        
        with st.expander("View Abstract"):
            st.markdown(paper['abstract'])
        
        pdf_url = paper['url'].replace('abs', 'pdf')
        st.markdown(f"[View Paper]({pdf_url})")
        
        col1, col2 = st.columns([3, 3])
        
        with col1:
            if st.button("📝 Q&A", key=f"qa_btn_{index}"):
                st.session_state.selected_paper = paper
                st.session_state.current_mode = "qa"
                paper_id = str(hash(paper['title']))
                rag_manager = RAGManager(working_dir=f"./lightrag_data/{paper_id}")
                st.session_state.current_rag = rag_manager

                ## Abstract for RAG
                #await rag_manager.process_paper(paper['abstract'], paper_id)
                
                # Full paper for RAG
                pdf_content = await fetch_pdf_content(pdf_url)
                await rag_manager.process_paper(pdf_content, paper_id)
        
        st.markdown("</div>", unsafe_allow_html=True)

async def display_qa_interface(paper, rag_manager, experience_level):
    """Display Q&A interface with experience-based prompting and graph visualization"""
    st.subheader(f"Q&A: {paper['title']}")
    
    col1, col2 = st.columns([3, 2])
    
    with col2:
        st.markdown("#### Knowledge Graph")
        if rag_manager.graph_path:
            with open(rag_manager.graph_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            components.html(html_content, height=500, width=500)
    
    with col1:
        prompt_templates = {
            "Beginner": "Explain in simple terms, avoiding technical jargon: ",
            "Intermediate": "Provide a balanced explanation with some technical details: ",
            "Advanced": "Give a detailed technical explanation: ",
            "Expert": "Provide an in-depth analysis with theoretical foundations: "
        }
        
        question = st.text_input("Your Question:")
        if st.button("Get Answer"):
            if question:
                with st.spinner("Analyzing paper..."):
                    context = f"""Experience level: {experience_level}
                    Please provide a {experience_level.lower()}-level answer to: {question}
                    {prompt_templates[experience_level]}"""
                    
                    try:
                        answer = await rag_manager.query_papers(
                            question, 
                            context,
                            mode="hybrid"
                        )
                        st.markdown(f"### Answer\n{answer}")
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")

async def async_search(recommender, search_query, search_method, experience_level=None):
    """Handle async search operations"""
    if search_method == "Quick Search (Titles)":
        return await recommender.recommend_by_titles(search_query)
    else:
        return await recommender.recommend_by_abstracts(
            search_query,
            experience_level=experience_level
        )

async def main():
    st.set_page_config(page_title="ResChat", layout="wide")
    
    init_session_state()
    
    st.title("ResChat: Research Paper Assistant")
    
    with st.sidebar:
        st.title("Search Preferences")
        
        experience_level = st.select_slider(
            "Experience Level",
            options=["Beginner", "Intermediate", "Advanced", "Expert"],
            value="Intermediate",
            help="Adjusts content complexity and technical detail"
        )
        
        search_method = st.radio(
            "Search Method",
            ["Quick Search (Titles)", "Deep Search (Abstracts)"],
            help="Quick Search matches titles, Deep Search analyzes full abstracts"
        )
        
        st.markdown("---")
        st.markdown("### About ResChat")
        st.info(
            "ResChat helps you discover and understand research papers using the power of Knowledge Graphs and AI."
            "tailored to your experience level and interests."
        )
    
    search_query = st.text_input(
        "Research Interest or Question",
        placeholder="Enter your research topic or specific question..."
    )
    
    if st.button("Search Papers", type="primary"):
        with st.spinner("Searching for relevant papers..."):
            try:
                recommendations = await async_search(
                    st.session_state.recommender,
                    search_query,
                    search_method,
                    experience_level
                )
                
                st.session_state.recommended_papers = recommendations
                
                if recommendations and "enhanced_query" in recommendations[0]:
                    st.success(f"Enhanced search query: {recommendations[0]['enhanced_query']}")
            
            except Exception as e:
                st.error(f"Search error: {str(e)}")
                raise e
    
    if st.session_state.recommended_papers:
        st.subheader("Recommended Papers")
        for i, paper in enumerate(st.session_state.recommended_papers):
            await create_paper_card(paper, i)
    
    if st.session_state.selected_paper and st.session_state.current_rag:
        if st.session_state.current_mode == "qa":
            await display_qa_interface(
                st.session_state.selected_paper,
                st.session_state.current_rag,
                experience_level
            )

if __name__ == "__main__":
    asyncio.run(main())