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
        
    def is_conditional_query(self, query: str) -> bool:
        """Use the LLM to classify if a query is conditional or hypothetical."""
        prompt = f"""Determine if this question is asking about a hypothetical or conditional scenario:
"{query}"

A conditional question asks what would happen IF a certain condition were true.
Examples of conditional questions:
- "If Kevin Tolson's foot had been objectively verified as cold to the touch, what would the nurse's response be?"
- "Assuming Hamden Heights North Park had been officially designated as a park, what process would be required?"
- "What symptoms might Juanita Frazier have exhibited if she had been awake when shot?"

Non-conditional questions:
- "What was Kevin Tolson's pain level at discharge?"
- "Who owns Hamden Heights North Park?"
- "What was the cause of Juanita Frazier's death?"

Is this a conditional/hypothetical question? Answer only Yes or No.
"""
        
        response = self.co.chat(
            message=prompt, 
            model="command-r-plus-08-2024", 
            temperature=0
        )
        return "yes" in response.text.lower()
        
    def get_relevant_chunks(self, query: str, limit: int = 10) -> List[Dict]:
        """Search the 'transcripts' index using Compass."""
        try:
            search_results = self.compass_client.search_chunks(
                index_name="compass-rev",  # updated index
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
        
    def plan_conditional_research(self, query: str) -> List[str]:
        """Plan research specifically for conditional/hypothetical questions."""
        prompt = f"""You are an expert analyzing transcripts for a CONDITIONAL or HYPOTHETICAL question:
"{query}"

This is a conditional query requiring us to understand:
1. The condition being proposed
2. What would happen under that condition
3. Any expert opinions or factual statements about such scenarios

Generate 4-5 search queries to gather all needed information:
- One query for understanding the factual details about the subject
- One query for understanding the condition itself
- One query for finding statements about consequences/outcomes of similar conditions
- One query for expert opinions or standards related to this scenario

Format each as a numbered line.
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

    def analyze_evidence(self, evidence: List[Dict], query: str, is_conditional: bool = False) -> Dict:
        """Check whether we have enough info to thoroughly answer the user's question."""
        evidence_text = "\n\n".join(e["snippet"] for e in evidence)
        
        if is_conditional:
            prompt = f"""We have the following transcript snippets related to this CONDITIONAL question: "{query}"

{evidence_text}

For conditional questions, we need to ensure we have:
1. Clear understanding of the condition/hypothesis
2. Expert statements about what would happen under such conditions
3. Relevant factual information to support the analysis

Do we have enough information to answer the question in detail? 
If not, what else do we need?

Respond in JSON:
{{
    "is_complete": true/false,
    "condition_clear": true/false,
    "consequence_clear": true/false,
    "gaps": ["gap1", "gap2"],
    "next_query": "some next search query"
}}
"""
        else:
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
        
    def synthesize_conditional_answer(self, evidence: List[Dict], query: str) -> str:
        """Generate answer specifically for conditional/hypothetical questions."""
        evidence_text = "\n\n".join(e["snippet"] for e in evidence)
        prompt = f"""
You are an expert analyst focusing on transcripts. The user asked this CONDITIONAL question: "{query}"

This requires analyzing a hypothetical scenario. Here are relevant transcript excerpts:
{evidence_text}

Focus on addressing:
1. Clearly identify the condition/hypothesis being proposed (the "if" part)
2. Explicitly state what would happen under that condition based on the evidence
3. Support your answer with direct testimony or expert opinions from the transcripts
4. Be specific about the consequences that would follow from the condition

Craft a comprehensive response that precisely addresses the conditional relationship. 
Begin by restating the hypothetical scenario, then systematically explain what would happen
according to the testimony or evidence provided.
"""
        response = self.co.chat(
            message=prompt,
            model="command-r-plus-08-2024",
            temperature=0.0
        )
        return response.text
        
    def validate_conditional_answer(self, answer: str, query: str) -> str:
        """Validates that a conditional answer properly addresses the hypothetical."""
        prompt = f"""
You are a quality checker for conditional/hypothetical answers. Review this answer to the conditional question:
"{query}"

ANSWER:
{answer}

Does this answer properly:
1. Identify the specific condition in the question?
2. Clearly state what would happen under that condition?
3. Support the conditional relationship with evidence?
4. Avoid being vague about the consequences?

If the answer fails any of these criteria, revise it to fix the issues.
If it's already good, return it unchanged.
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
        
        # Determine if this is a conditional query
        is_conditional = self.is_conditional_query(query)
        
        # Plan research steps based on query type
        if is_conditional:
            planned_queries = self.plan_conditional_research(query)
        else:
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
            
            # Analyze with appropriate method
            analysis = self.analyze_evidence(self.collected_evidence, query, is_conditional)
            step_count += 1
            
            if analysis.get("is_complete", False):
                break
            if step_count >= max_steps:
                break
                
            if analysis.get("next_query"):
                planned_queries.append(analysis["next_query"])
        
        # Generate answer with appropriate method based on query type
        if is_conditional:
            final_answer = self.synthesize_conditional_answer(self.collected_evidence, query)
            # Additional validation for conditional answers
            final_answer = self.validate_conditional_answer(final_answer, query)
        else:
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
