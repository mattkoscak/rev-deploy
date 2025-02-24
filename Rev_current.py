import streamlit as st
import hmac
import os
import json
from typing import List, Dict, Optional
import cohere
from cohere.client import Client as CohereClient
from cohere.compass.clients.compass import CompassClient

# -----------------------------------------
# Global Password Protection (as you have)
# -----------------------------------------
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

# -----------------------------------------
# Enhanced TranscriptRAGAgent Class
# -----------------------------------------
class TranscriptRAGAgent:
    """
    A RAG-based agent leveraging Compass for retrieval and Cohere for LLM.
    Incorporates:
    - Multi-step / iterative search with partial query expansion
    - Confidence-based chunk filtering
    - Strict anti-hallucination prompts
    - Final answer synthesis with optional concise summarization
    """

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

        self.research_steps = []
        self.collected_evidence = []
        self.confidence_threshold = 0.0  # Adjust if you want to filter low-scoring hits

    # ---------------
    # 1) Hybrid Search
    #    (Compass presumably does this behind the scenes, but we can re-rank or filter.)
    # ---------------
    def get_relevant_chunks(self, query: str, top_k: int = 8) -> List[Dict]:
        """
        Search the 'compass-rev' index using Compass. Return a list of chunk dicts:
        [
          {
            "title": "doc_0",
            "snippet": "...",
            "score": 0.88
          },
          ...
        ]
        """
        try:
            search_results = self.compass_client.search_chunks(
                index_name="compass-rev",
                query=query,
                top_k=top_k
            )
            documents = []
            if search_results.hits:
                for idx, hit in enumerate(search_results.hits):
                    text = hit.content.get("text", "")
                    score = hit.score if hasattr(hit, "score") else 1.0
                    # Only add if above a certain threshold
                    if score >= self.confidence_threshold:
                        documents.append({
                            "title": f"doc_{idx}",
                            "snippet": text,
                            "score": score
                        })
            return documents
        except Exception as e:
            st.error(f"Error retrieving documents: {e}")
            return []

    # ---------------
    # 2) Query Decomposition & Expansion
    #    - Very simple approach: we ask Cohere to suggest expansions or sub-queries
    #    - We can do multi-step searches
    # ---------------
    def plan_research(self, user_query: str) -> List[str]:
        """
        Use LLM to generate possible expansions/sub-queries.
        We'll do a short, simplistic prompt to get ~3 sub-queries.
        If none are returned, fallback to the user_query.
        """
        # System instructions: do not hallucinate, generate short expansions.
        system_message = (
            "You are a helpful assistant that breaks down complex questions into 2-5 smaller search queries. "
            "These queries should collectively gather all relevant information from transcripts. "
            "Keep them short and focused. If the user query is already simple, just return it unchanged."
        )
        prompt = (
            f"{system_message}\n\n"
            f"User Query: {user_query}\n\n"
            "Provide 2-5 bullet points, each a short search query or sub-question. Example:\n"
            "1) 'Key events in the 2025 all-hands'\n"
            "2) 'Revenue model changes'\n"
        )
        response = self.co.chat(
            message=prompt,
            model="command-r-plus-08-2024",
            temperature=0.2
        )

        # Attempt to parse steps from the response
        lines = response.text.strip().split("\n")
        steps = []
        for line in lines:
            line_stripped = line.strip()
            # Heuristic: lines that start with a digit or dash
            if line_stripped and (line_stripped[0].isdigit() or line_stripped.startswith("-")):
                # remove numbering for cleanliness
                cleaned = line_stripped.lstrip("0123456789.-) ")
                steps.append(cleaned)

        if not steps:
            steps = [user_query]  # fallback

        # Sanity limit to keep steps short
        steps = steps[:5]
        return steps

    # ---------------
    # 3) Checking Gaps & Additional Needed Queries
    #    - We'll ask the LLM if the current evidence is enough or if we need more queries
    # ---------------
    def analyze_evidence(self, query: str, evidence: List[Dict]) -> Dict:
        """
        Ask LLM if the info is complete enough. If not, LLM suggests next query or any identified gaps.
        Returns a dict:
        {
          "is_complete": bool,
          "gaps": ["gap1", "gap2"],
          "next_query": "some new query or None"
        }
        """
        # Merge snippet text
        evidence_text = ""
        for i, e in enumerate(evidence):
            snippet = e["snippet"][:500]  # limit size to reduce token usage
            evidence_text += f"Snippet {i+1} (score={e['score']}):\n{snippet}\n\n"

        # Prompt
        sys_prompt = (
            "You are a world-class analyst. You only rely on the provided transcript excerpts "
            "and do NOT hallucinate. If not enough info is present, you must say so."
        )
        prompt = (
            f"{sys_prompt}\n\n"
            f"User query: '{query}'\n\n"
            f"Relevant transcript excerpts:\n{evidence_text}\n\n"
            "Based on these excerpts, do we have enough info to provide a detailed, correct answer? "
            "If not, suggest what is missing or a new search query to find it.\n\n"
            "Respond in valid JSON:\n"
            "{\n"
            "  \"is_complete\": true/false,\n"
            "  \"gaps\": [\"gap1\", \"gap2\"],\n"
            "  \"next_query\": \"some next search query or null\"\n"
            "}\n"
        )
        response = self.co.chat(
            message=prompt,
            model="command-r-plus-08-2024",
            temperature=0.2
        )

        # Attempt to parse JSON. If it fails, default to complete = True
        try:
            parsed = json.loads(response.text)
            # basic structure check
            if not isinstance(parsed, dict):
                raise ValueError("Response not a JSON object")
            if "is_complete" not in parsed:
                parsed["is_complete"] = True
            if "gaps" not in parsed:
                parsed["gaps"] = []
            if "next_query" not in parsed:
                parsed["next_query"] = None
            return parsed
        except:
            return {
                "is_complete": True,
                "gaps": [],
                "next_query": None
            }

    # ---------------
    # 4) Final Answer Synthesis
    #    - We ensure it doesn't hallucinate by instructing it to only use provided snippets
    #    - We embed references to doc_i where relevant
    # ---------------
    def synthesize_answer(self, query: str, evidence: List[Dict], fallback_if_empty: bool = True) -> str:
        """
        Generate final comprehensive answer from transcript snippets.
        If no evidence is present, optionally fallback to a "no info" message.
        """
        if not evidence and fallback_if_empty:
            return (
                "I'm sorry, but I couldn't find any relevant information in the transcripts to answer this question."
            )

        # Combine top chunks for final synthesis
        # In practice, you could limit how many total tokens pass into the prompt
        evidence_text = ""
        for i, e in enumerate(evidence):
            snippet = e["snippet"].replace("\n", " ")
            evidence_text += f"[Source doc_{i}, score={e['score']:.2f}]: {snippet}\n\n"

        # Strict instructions to avoid hallucination
        sys_prompt = (
            "You are a careful, fact-based assistant that ONLY uses the provided transcript snippets to answer. "
            "If the information is not in the snippets, say 'I do not have enough information from the transcripts'."
        )

        # The prompt
        prompt = f"""
{sys_prompt}

User asked:
"{query}"

Below are relevant transcript excerpts:
{evidence_text}

Task:
Provide a thorough, cohesive answer to the user's query, referencing [Source doc_i] whenever you pull information from that snippet.
If there's insufficient info, say: "I do not have enough information from the transcripts."

Final Answer:
"""
        response = self.co.chat(
            message=prompt,
            model="command-r-plus-08-2024",
            temperature=0.0
        )
        return response.text.strip()

    # ---------------
    # 5) Multi-step Research Orchestrator
    #    - Uses plan_research() -> get_relevant_chunks() -> analyze_evidence() -> possible next steps
    # ---------------
    def research(self, user_query: str, max_rounds: int = 3, top_k: int = 8) -> Dict:
        """
        Orchestrates a multi-step research approach:
          1) Generate sub-queries or expansions
          2) For each sub-query, retrieve top_k chunks
          3) Accumulate evidence, check if we can answer
          4) Possibly refine or do next query if not complete
        """
        self.research_steps = []
        self.collected_evidence = []

        # 1) Sub-queries
        planned_queries = self.plan_research(user_query)

        round_count = 0
        for sq in planned_queries:
            if round_count >= max_rounds:
                break

            # Save the step
            self.research_steps.append({
                "step": round_count + 1,
                "sub_query": sq
            })

            # 2) Retrieve new evidence
            new_evidence = self.get_relevant_chunks(sq, top_k=top_k)

            # 3) Add to our global evidence
            self.collected_evidence.extend(new_evidence)

            # 4) Analyze if we have enough
            analysis = self.analyze_evidence(user_query, self.collected_evidence)
            round_count += 1

            if analysis["is_complete"]:
                # No further refinement needed
                break
            else:
                # If we have next_query, we append it to planned_queries
                if analysis["next_query"]:
                    planned_queries.append(analysis["next_query"])

        # Final answer
        final_answer = self.synthesize_answer(user_query, self.collected_evidence)
        return {
            "query": user_query,
            "steps": self.research_steps,
            "evidence": self.collected_evidence,
            "answer": final_answer
        }

    # ---------------
    # 6) Optionally produce a concise version of the final answer
    # ---------------
    def make_concise(self, detailed_answer: str) -> str:
        """
        Use Cohere to rewrite the final answer into a concise version.
        """
        if not detailed_answer.strip():
            return detailed_answer

        sys_prompt = (
            "You are a world-class summarizer. Convert the user-provided answer into a concise version "
            "using only the facts present. Do not add new information."
        )
        prompt = f"""
{sys_prompt}

Detailed Answer:
\"\"\"{detailed_answer}\"\"\"

Now rewrite it in concise form without losing key facts:
"""
        response = self.co.chat(
            message=prompt,
            model="command-r-plus-08-2024",
            temperature=0.0
        )
        return response.text.strip()

# -----------------------------------------
# Streamlit UI
# -----------------------------------------
st.set_page_config(
    page_title="Transcript RAG - Advanced Demo",
    page_icon="🔍",
    layout="wide"
)

# Inline CSS
st.markdown("""
<style>
.research-step {
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.5rem;
    border-radius: 4px;
    background-color: #f8f9fa;
}
.evidence-box {
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    border-radius: 4px;
    background-color: #e9ecef;
    font-size: 0.9rem;
}
.answer-section {
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 8px;
    background-color: #ffffff;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Initialize agent if not in session
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
    st.title("Transcript RAG - Settings")
    st.write("Adjust the search parameters and output style.")
    top_k = st.slider("Number of chunks to retrieve per sub-query:", 3, 15, 8)
    max_rounds = st.slider("Max multi-step rounds:", 1, 5, 3)
    st.write("---")
    concise_toggle = st.checkbox("Generate a concise final answer", value=False)
    st.write("---")
    threshold = st.slider("Confidence threshold (filter out low-scoring chunks):", 0.0, 1.0, 0.0, 0.05)
    st.info("Chunks with a retrieval score below this threshold will be excluded.")

    # Save user-chosen threshold
    st.session_state.agent.confidence_threshold = threshold

st.title("Transcript RAG Q&A (Improved)")

# Input
query = st.text_area("Enter your transcript-based question:", height=100)

if st.button("Submit"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Performing multi-step retrieval and answer synthesis..."):
            result = st.session_state.agent.research(
                user_query=query,
                max_rounds=max_rounds,
                top_k=top_k
            )
            final_answer = result["answer"]
            
            # Optionally re-summarize for a concise version
            if concise_toggle:
                with st.spinner("Generating concise version..."):
                    final_answer = st.session_state.agent.make_concise(final_answer)

        # Display the results
        st.subheader("Answer")
        st.write(final_answer)

        # Expand to show retrieval steps
        with st.expander("Research Steps and Evidence"):
            st.markdown("#### Steps")
            for step_data in result["steps"]:
                st.markdown(f"<div class='research-step'><strong>Step {step_data['step']}</strong>: {step_data['sub_query']}</div>", unsafe_allow_html=True)

            st.markdown("#### Evidence Used")
            for idx, ev in enumerate(result["evidence"]):
                st.markdown(
                    f"<div class='evidence-box'><strong>Source doc_{idx}</strong> "
                    f"(Score: {ev['score']:.2f})<br>{ev['snippet']}</div>",
                    unsafe_allow_html=True
                )

        # If the final answer indicates insufficient info, highlight that
        if "I do not have enough information" in final_answer:
            st.warning("⚠️ The system did not find enough evidence in the transcripts to fully answer.")


# End of code
