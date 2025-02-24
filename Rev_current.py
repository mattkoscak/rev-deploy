import hmac
import streamlit as st
import os
import json
import time
from typing import List, Dict, Optional
import cohere
from cohere.client import Client as CohereClient
from cohere.compass.clients.compass import CompassClient

# --- Global Password Protection ---
def check_password():
    """
    Returns True if the user enters the correct global password.
    The expected password is stored in st.secrets["password"].
    """
    def password_entered():
        # Compare user-entered password with the stored secret
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            # Remove password from session state for security
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input("Password", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Password incorrect")
    return st.session_state.get("password_correct", False)

if not check_password():
    st.stop()
# --- End Global Password Protection ---


# -------------------- TranscriptRAGAgent Class --------------------
class TranscriptRAGAgent:
    def __init__(self, compass_url: str, compass_token: str, cohere_api_key: str):
        # Initialize API clients
        self.co = CohereClient(
            api_key=cohere_api_key,
            client_name="transcript-rag-agent"
        )
        self.compass_client = CompassClient(
            index_url=compass_url,
            bearer_token=compass_token
        )
        # Tracking research state
        self.research_steps = []
        self.collected_evidence = []
        
    def get_relevant_chunks(self, query: str, limit: int = 10) -> List[Dict]:
        """Search the 'transcripts' index using Compass."""
        try:
            search_results = self.compass_client.search_chunks(
                index_name="transcripts",  # updated index
                query=query,
                top_k=limit
            )
            documents = []
            if search_results.hits:
                for idx, hit in enumerate(search_results.hits):
                    text = hit.content.get("text", "")
                    documents.append({
                        "title": f"doc_{idx}",
                        "snippet": text
                    })
            return documents
        except Exception as e:
            st.error(f"Error retrieving documents: {e}")
            return []

    def plan_research(self, query: str) -> List[str]:
        """Plan a multi-step approach to gather thorough insights from transcripts."""
        prompt = f"""You are an expert analyzing transcripts. The user wants to explore this question:
"{query}"

Generate 3-5 short 'search queries' or 'research steps' that will gather all needed information.
Example:
1. "Azure revenue over time"
2. "Factors impacting cloud growth"
"""
        response = self.co.chat(
            message=prompt,
            model="command-r-plus-08-2024",
            temperature=0.2
        )
        
        steps = [
            line.strip() 
            for line in response.text.split("\n") 
            if line.strip() and line[0].isdigit()
        ]
        # Fallback if no steps extracted
        if not steps:
            steps = [query]
        return steps

    def analyze_evidence(self, evidence: List[Dict], query: str) -> Dict:
        """Check whether we have enough info to thoroughly answer the user's question."""
        evidence_text = "\n\n".join(e["snippet"] for e in evidence)
        
        prompt = f"""We have the following transcript snippets related to: "{query}"

{evidence_text}

Do we have enough information to answer the question in detail? 
If not, what else do we need?

Respond in JSON:
{{
    "is_complete": true/false,
    "gaps": ["gap1", "gap2"],
    "next_query": "some next search query"
}}
"""
        response = self.co.chat(
            message=prompt,
            model="command-r-plus-08-2024",
            temperature=0.1
        )
        
        try:
            return json.loads(response.text)
        except:
            return {
                "is_complete": True,
                "gaps": [],
                "next_query": None
            }

    def synthesize_answer(self, evidence: List[Dict], query: str) -> str:
        """Generate final comprehensive answer from transcript snippets."""
        evidence_text = "\n\n".join(e["snippet"] for e in evidence)
        prompt = f"""
You are a top-tier analyst focusing on transcripts. The user asked: "{query}"
Here are relevant transcript excerpts:
{evidence_text}

Craft a thorough, cohesive summary or answer addressing the user's question. 
Use examples, references (like [doc_0]) if needed. Maintain a clear and concise style.
"""
        response = self.co.chat(
            message=prompt,
            model="command-r-plus-08-2024",
            temperature=0.0
        )
        return response.text

    def research(self, query: str, max_steps: int = 5) -> Dict:
        """
        Execute the multi-step approach to find and synthesize an answer for a 
        transcript-based question.
        """
        self.research_steps = []
        self.collected_evidence = []
        
        planned_queries = self.plan_research(query)
        
        step_count = 0
        for search_query in planned_queries:
            if step_count >= max_steps:
                break
                
            self.research_steps.append({
                "step": step_count + 1,
                "action": "search",
                "query": search_query
            })
            
            # Search
            new_evidence = self.get_relevant_chunks(search_query)
            self.collected_evidence.extend(new_evidence)
            
            # Analyze
            analysis = self.analyze_evidence(self.collected_evidence, query)
            step_count += 1
            
            if analysis["is_complete"]:
                break
            if step_count >= max_steps:
                break
                
            if analysis["next_query"]:
                planned_queries.append(analysis["next_query"])
        
        final_answer = self.synthesize_answer(self.collected_evidence, query)
        
        return {
            "query": query,
            "steps": self.research_steps,
            "evidence": self.collected_evidence,
            "answer": final_answer
        }

# -------------------- End TranscriptRAGAgent Class --------------------


# -------------------- Streamlit UI Code --------------------
st.set_page_config(
    page_title="Multi file insights with Rev",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
.research-step {
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 4px;
    background-color: #f8f9fa;
}
.evidence-box {
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 4px;
    background-color: #e9ecef;
}
.answer-section {
    padding: 2rem;
    margin: 1rem 0;
    border-radius: 8px;
    background-color: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

if 'agent' not in st.session_state:
    required_vars = ["COHERE_API_KEY", "COMPASS_TOKEN", "COMPASS_URL"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        st.error(f"Missing environment variables: {', '.join(missing_vars)}")
        st.stop()
    
    st.session_state.agent = TranscriptRAGAgent(
        compass_url=os.environ["COMPASS_URL"],
        compass_token=os.environ["COMPASS_TOKEN"],
        cohere_api_key=os.environ["COHERE_API_KEY"]
    )

# Sidebar
with st.sidebar:
    st.title("Cohere powered RAG")
    st.write("""
    Provide a question about your transcripts,
    and I'll retrieve and summarize relevant insights.
    """)
    max_steps = 5

# Main interface
st.title("Multi file insights with Rev")
query = st.text_area("Enter your transcript-based question:", height=100)

if st.button("Submit"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Gathering insights..."):
            result = st.session_state.agent.research(query, max_steps=max_steps)
            
            # Display final answer only
            st.subheader("Answer")
            st.markdown(result["answer"])
            
            # Optionally show sources
            with st.expander("View Sources"):
                for idx, evidence in enumerate(result["evidence"]):
                    st.markdown(f"**Source {idx + 1}:**")
                    st.markdown(evidence["snippet"])
                    st.markdown("---")
