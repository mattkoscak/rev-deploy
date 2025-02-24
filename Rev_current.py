import os
import json
import time
from typing import List, Dict, Optional
import cohere
from cohere.client import Client as CohereClient
from cohere.compass.clients.compass import CompassClient
import streamlit as st

import streamlit as st
import streamlit_authenticator as stauth

# Configure credentials (change these as needed)
names = ['rev1']
usernames = ['rev1']
# Replace 'YourPasswordHere' with your desired password.
hashed_passwords = stauth.Hasher(['rev1']).generate()

authenticator = stauth.Authenticate(
    {'names': names, 'usernames': usernames, 'passwords': hashed_passwords},
    'app_cookie', 'signature_key', cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login('Login', 'main')

if not authentication_status:
    st.stop()  # Prevents the app from running if not authenticated


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
        prompt = f"""You are an expert analyst of transcript data. The user asked:
"{query}"
Generate 3-5 short, focused research steps (or search queries) to cover all relevant aspects of this question.
Examples:
1. "Key discussion points on family health from the transcript"
2. "Mentions of physical activity and shared meals in the discussion"
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
        
        prompt = f"""We have the following transcript excerpts related to the question:
"{query}"

{evidence_text}

Based on these excerpts, do we have enough detailed information to provide a comprehensive answer?
If not, list what additional details or angles are needed.
Respond in JSON in the following format:
{{
    "is_complete": true/false,
    "gaps": ["gap1", "gap2"],
    "next_query": "a focused follow-up query"
}}
"""
        response = self.co.chat(
            message=prompt,
            model="command-r-plus-08-2024",
            temperature=0.1
        )
        
        try:
            return json.loads(response.text)
        except Exception as e:
            return {
                "is_complete": True,
                "gaps": [],
                "next_query": None
            }

    def synthesize_answer(self, evidence: List[Dict], query: str) -> str:
        """Generate final comprehensive answer from transcript excerpts."""
        evidence_text = "\n\n".join(e["snippet"] for e in evidence)

        # We include prompt engineering here to encourage answers that reference the focus group discussion.
        prompt = f"""
You are a top-tier analyst specializing in transcript data. The user asked:
"{query}"
Below are relevant transcript excerpts:
{evidence_text}

Based on these excerpts, craft a detailed and cohesive answer that explains the key points. 
The answer should begin by referencing the discussion (e.g., "The discussion highlights that ...") and focus on explaining how the described actions (such as joint physical activities and shared meals) promote overall well-being.
Avoid extraneous details like financial or environmental factors unless they are essential. 
Include references to sources when relevant (e.g., [doc_0]).
"""
        response = self.co.chat(
            message=prompt,
            model="command-r-plus-08-2024",
            temperature=0.0
        )
        
        return response.text

    def research(self, query: str, max_steps: int = 5) -> Dict:
        """
        Execute a multi-step approach to synthesize an answer for a transcript-based question.
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
            
            # Search for relevant transcript chunks
            new_evidence = self.get_relevant_chunks(search_query)
            self.collected_evidence.extend(new_evidence)
            
            # Analyze if we have enough evidence
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

# -------------------- Streamlit UI Code --------------------
st.set_page_config(
    page_title="Multi file insights with Rev",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
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

# Initialize agent in session state if not present
if 'agent' not in st.session_state:
    required_vars = [
        "COHERE_API_KEY",
        "COMPASS_TOKEN",
        "COMPASS_URL"
    ]
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        st.error(f"Missing environment variables: {', '.join(missing_vars)}")
        st.stop()
    
    st.session_state.agent = TranscriptRAGAgent(
        compass_url=os.environ["COMPASS_URL"],
        compass_token=os.environ["COMPASS_TOKEN"],
        cohere_api_key=os.environ["COHERE_API_KEY"]
    )

# Sidebar UI
with st.sidebar:
    st.title("Cohere powered RAG")
    st.write("""
    Provide a question about your transcripts,
    and I'll retrieve and summarize relevant insights.
    """)
    max_steps = 5
    # Toggle to enable concise answer generation
    concise_toggle = st.checkbox("Generate a concise answer", value=False)

# Main interface
st.title("Multi file insights with Rev")
query = st.text_area("Enter your transcript-based question:", height=100)

if st.button("Submit"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Gathering insights..."):
            result = st.session_state.agent.research(query, max_steps=max_steps)
            final_answer = result["answer"]
            
            # If concise toggle is enabled, transform the answer via additional prompt engineering
            if concise_toggle:
                concise_prompt = f"""
Rewrite the following detailed answer into a concise, well-rounded summary.
Ensure the summary starts by referencing the focus group discussion, for example: "The discussion highlights that ..."
Focus on how the actions described in the answer promote overall well-being.
Do not include extraneous details such as mentions of financial resources or environmental factors unless directly relevant.
Do not include any Q: or A: labels; only produce the final answer text.

Detailed answer:
"{final_answer}"
"""
                response = st.session_state.agent.co.chat(
                    message=concise_prompt,
                    model="command-r-plus-08-2024",
                    temperature=0.0
                )
                final_answer = response.text.strip()
            
            # Display only the final answer text
            st.subheader("Answer")
            st.markdown(final_answer)
            
            # Optionally show sources if needed
            with st.expander("View Sources"):
                for idx, evidence in enumerate(result["evidence"]):
                    st.markdown(f"**Source {idx + 1}:**")
                    st.markdown(evidence["snippet"])
                    st.markdown("---")
