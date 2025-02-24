import hmac
import streamlit as st
import os
import json
from typing import List, Dict

# --- Global Password Protection ---
def check_password():
    """
    Returns True if the user enters the correct global password.
    The expected password is stored in st.secrets["password"].
    """
    def password_entered():
        # Compare user-entered password with the stored secret using secure comparison
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            # Remove the password from session state for security
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


# --- TranscriptRAGAgent Class with Modular Steps ---
import cohere
from cohere.client import Client as CohereClient
from cohere.compass.clients.compass import CompassClient

class TranscriptRAGAgent:
    def __init__(self, compass_url: str, compass_token: str, cohere_api_key: str):
        # Initialize API clients
        self.co = CohereClient(api_key=cohere_api_key, client_name="transcript-rag-agent")
        self.compass_client = CompassClient(index_url=compass_url, bearer_token=compass_token)
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Break down the original query into sub-questions that will help gather all needed evidence.
        Uses chain-of-thought prompting with few-shot examples.
        """
        decomposition_prompt = f"""
You are an expert in transcript analysis. Break down the following question into a list of specific sub-questions that, if answered, would comprehensively cover the topic.
Question: "{query}"
Examples:
- For "What challenges did the interview highlight?" list:
    1. What technology challenges were mentioned?
    2. How are operational issues discussed?
    3. What financial impacts were noted?
Now, list the sub-questions for the above query.
"""
        response = self.co.chat(
            message=decomposition_prompt, 
            model="command-r-plus-08-2024", 
            temperature=0.2
        )
        # Expect each sub-question to be on a new line starting with a number
        sub_questions = [line.strip() for line in response.text.split("\n") if line.strip() and line[0].isdigit()]
        # Fallback: if no sub-questions are produced, use the original query
        return sub_questions if sub_questions else [query]
    
    def retrieve_evidence(self, sub_queries: List[str], limit: int = 10) -> List[Dict]:
        """
        For each sub-question, query Compass to retrieve transcript chunks.
        """
        evidence = []
        for sub_query in sub_queries:
            try:
                search_results = self.compass_client.search_chunks(
                    index_name="compass-rev",
                    query=sub_query,
                    top_k=limit
                )
                if search_results.hits:
                    for idx, hit in enumerate(search_results.hits):
                        evidence.append({
                            "title": f"doc_{len(evidence)}",
                            "snippet": hit.content.get("text", "")
                        })
            except Exception as e:
                # Log error and continue with next sub-query
                st.error(f"Error retrieving evidence for '{sub_query}': {e}")
        return evidence
    
    def aggregate_evidence(self, evidence: List[Dict]) -> str:
        """
        Aggregate evidence by joining all transcript snippets.
        (This can be enhanced with summarization if needed.)
        """
        return "\n\n".join(e["snippet"] for e in evidence)
    
    def synthesize_final_answer(self, aggregated_evidence: str, query: str) -> str:
        """
        Synthesize a final answer using the aggregated evidence and original query.
        """
        synthesis_prompt = f"""
You are a top-tier analyst specializing in transcript data. Given the following aggregated transcript excerpts and the original question, craft a detailed and cohesive answer. 
Your answer should begin by referencing the discussion (e.g., "The discussion highlights that ...") and explain the key insights.
Reference evidence when appropriate (e.g., [doc_0]).

Original Question: "{query}"
Aggregated Evidence:
{aggregated_evidence}

Answer:
"""
        response = self.co.chat(
            message=synthesis_prompt, 
            model="command-r-plus-08-2024", 
            temperature=0.0
        )
        return response.text
    
    def research(self, query: str, max_steps: int = 5) -> Dict:
        """
        Orchestrate the research process:
        1. Decompose the query into sub-questions.
        2. Retrieve evidence for each sub-question.
        3. Aggregate all evidence.
        4. Synthesize the final answer.
        """
        # Step 1: Decompose query
        sub_queries = self.decompose_query(query)
        
        # Step 2: Retrieve evidence for each sub-question
        evidence = self.retrieve_evidence(sub_queries)
        
        # Step 3: Aggregate evidence
        aggregated_text = self.aggregate_evidence(evidence)
        
        # Step 4: Synthesize final answer
        final_answer = self.synthesize_final_answer(aggregated_text, query)
        
        return {
            "query": query,
            "sub_queries": sub_queries,
            "evidence": evidence,
            "aggregated_text": aggregated_text,
            "answer": final_answer
        }

# --- End TranscriptRAGAgent Class ---


# --- Streamlit UI Code ---
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

# Ensure the required environment variables are present
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

with st.sidebar:
    st.title("Cohere powered RAG")
    st.write("""
    Provide a question about your transcripts,
    and I'll retrieve and summarize relevant insights.
    """)
    max_steps = 5
    concise_toggle = st.checkbox("Generate a concise answer", value=False)

st.title("Multi file insights with Rev")
query = st.text_area("Enter your transcript-based question:", height=100)

if st.button("Submit"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Gathering insights..."):
            result = st.session_state.agent.research(query, max_steps=max_steps)
            final_answer = result["answer"]
            
            # Optionally, generate a concise version of the answer
            if concise_toggle:
                concise_prompt = f"""
Rewrite the following detailed answer into a concise, well-rounded summary.
Ensure the summary starts by referencing the focus group discussion (e.g., "The discussion highlights that ...") and focuses on how the actions promote overall well-being.
Avoid extraneous details such as financial or environmental factors unless directly relevant.
Do not include any labels (like Q: or A:).

Detailed answer:
"{final_answer}"
"""
                response = st.session_state.agent.co.chat(
                    message=concise_prompt,
                    model="command-r-plus-08-2024",
                    temperature=0.0
                )
                final_answer = response.text.strip()
            
            st.subheader("Answer")
            st.markdown(final_answer)
            
            with st.expander("View Sources"):
                for idx, evidence in enumerate(result["evidence"]):
                    st.markdown(f"**Source {idx + 1}:**")
                    st.markdown(evidence["snippet"])
                    st.markdown("---")
