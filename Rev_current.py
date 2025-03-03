import hmac
import streamlit as st
import os
import json
import time
import re
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
        
    def get_relevant_chunks(self, query: str, limit: int = 25) -> List[Dict]:
        """
        Search the 'transcripts' index using Compass.
        Returns more chunks for better coverage of broad queries.
        """
        try:
            search_results = self.compass_client.search_chunks(
                index_name="rev-custom-contextual-chunks",
                query=query,
                top_k=limit
            )
            documents = []
            if search_results.hits:
                for idx, hit in enumerate(search_results.hits):
                    text = hit.content.get("text", "")
                    
                    # Try to extract source information safely
                    source_filename = f"document_{idx}"
                    # Check if there's document information in the content
                    if hasattr(hit, 'document_id'):
                        source_filename = hit.document_id
                    
                    documents.append({
                        "title": f"doc_{idx}",
                        "source": source_filename,
                        "snippet": text
                    })
            return documents
        except Exception as e:
            st.error(f"Error retrieving documents: {e}")
            return []

    def plan_research(self, query: str) -> List[str]:
        """
        Plan a multi-step approach to gather thorough insights from transcripts.
        Enhanced to generate more specific and targeted search queries.
        """
        prompt = f"""You are an expert analyzing transcripts. The user wants to explore this question:
"{query}"

Your task is to break down this question into 4-6 specific search queries that together will provide comprehensive coverage.

For broad questions that involve multiple entities, time periods, or concepts, create separate queries for each component.
For questions about specific projects or concepts that might be ambiguous, create queries that help distinguish them from similar terms.

Generate specific queries that:
1. Cover different aspects of the main question
2. Help disambiguate between potentially confusing concepts
3. Target specific time periods, people, or projects mentioned
4. Gather comprehensive information across multiple documents

Return ONLY numbered search queries, one per line:
"""
        response = self.co.chat(
            message=prompt,
            model="command-r-plus-08-2024",
            temperature=0.2
        )
        
        steps = [
            line.strip().strip('"\'') 
            for line in response.text.split("\n") 
            if line.strip() and (line[0].isdigit() or re.match(r'^"', line.strip()))
        ]
        # Clean up the steps to remove numbers and quotes
        steps = [re.sub(r'^\d+\.\s*', '', step).strip('"\'') for step in steps if step]
        
        # Fallback if no steps extracted
        if not steps:
            steps = [query]
        return steps

    def analyze_evidence(self, evidence: List[Dict], query: str) -> Dict:
        """
        Check whether we have enough info to thoroughly answer the user's question.
        Enhanced to better identify gaps in information.
        """
        evidence_text = "\n\n".join([
            f"[Source: {e.get('source', 'document')}]\n{e['snippet']}" 
            for e in evidence
        ])
        
        prompt = f"""We have the following transcript snippets related to: "{query}"

{evidence_text}

Your task is to carefully analyze whether we have sufficient, relevant, and comprehensive information to answer the original question.

Consider:
1. Do we have relevant information from multiple perspectives?
2. Is there information about all entities, projects, or concepts mentioned in the question?
3. For time-based questions, do we have data covering the entire requested time period?
4. For questions about specific concepts, do we have clear definitions that disambiguate them from similar concepts?

Respond in JSON:
{{
    "is_complete": true/false,
    "gaps": ["specific gap 1", "specific gap 2"],
    "next_query": "specific search query to fill the most important gap"
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
        """
        Generate final comprehensive answer from transcript snippets.
        Uses a simpler approach that gives the model more freedom to organize the response.
        """
        evidence_text = "\n\n".join([
            f"[Source: {e.get('source', 'document')}]\n{e['snippet']}" 
            for e in evidence
        ])
        
        prompt = f"""You are a top-tier analyst focusing on transcripts. The user asked: "{query}"

Here are relevant transcript excerpts:
{evidence_text}

Craft a thorough, cohesive summary or answer addressing the user's question. 
Organize information logically and use appropriate formatting to make your answer clear and readable.
Do not include source citations in your answer.

When discussing projects or activities:
- Group related items together into meaningful categories
- Consider using bullet points for lists when appropriate
- Use bold formatting to highlight key projects or themes
- Acknowledge any limitations or inconsistencies in the information

Maintain a clear and concise style throughout your response.
"""
        response = self.co.chat(
            message=prompt,
            model="command-r-plus-08-2024",
            temperature=0.0
        )
        
        # Add a "Sources" section at the end with all unique sources
        unique_sources = list(set(e.get('source', 'document') for e in evidence))
        sources_text = "\n\n## Sources\n" + "\n".join([f"{source}" for source in unique_sources])
        
        return response.text + sources_text

    def research(self, query: str, max_steps: int = 5) -> Dict:
        """
        Execute the multi-step approach to find and synthesize an answer for a 
        transcript-based question.
        Enhanced to gather more comprehensive results.
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
            
            # Search with increased limit for more comprehensive results
            new_evidence = self.get_relevant_chunks(search_query, limit=25)
            
            # Only add unique evidence (avoid duplicates)
            existing_snippets = {e["snippet"] for e in self.collected_evidence}
            unique_new_evidence = [e for e in new_evidence if e["snippet"] not in existing_snippets]
            
            self.collected_evidence.extend(unique_new_evidence)
            
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
    page_title="Multi file insights",
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
    st.title("Version B")
    max_steps = 5

# Main interface
st.title("Multi file insights with Rev")
query = st.text_area("Enter your question about the transcripts below...:", height=100)

if st.button("Submit"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Gathering insights..."):
            result = st.session_state.agent.research(query, max_steps=max_steps)
            
            # Display final answer only
            st.subheader("Answer")
            st.markdown(result["answer"])