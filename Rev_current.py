import streamlit as st
import hmac
import os
import json
from typing import List, Dict, Optional
import cohere
from cohere.client import Client as CohereClient
from cohere.compass.clients.compass import CompassClient

# -----------------------------------------
# Global Password Protection (if desired)
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

# Comment out if you don't want password protection at all
if not check_password():
    st.stop()


# -----------------------------------------
# Adjustable Defaults (hidden from the user)
# -----------------------------------------
TOP_K = 8                   # How many chunks to retrieve per sub-query
MAX_ROUNDS = 3             # Max multi-step sub-queries
CONFIDENCE_THRESHOLD = 0.0  # Filter out low-scoring retrieval hits (0.0 means no filtering)

# -----------------------------------------
# TranscriptRAGAgent Class
# -----------------------------------------
class TranscriptRAGAgent:
    """
    A RAG-based agent leveraging Compass for retrieval and Cohere for LLM.
    Incorporates:
    - Multi-step / iterative search with partial query expansion
    - Confidence-based chunk filtering
    - Strict anti-hallucination prompts
    - Final answer synthesis
    """

    def __init__(self, compass_url: str, compass_token: str, cohere_api_key: str):
        # Initialize API clients
        try:
            self.co = CohereClient(
                api_key=cohere_api_key,
                client_name="transcript-rag-agent"
            )
        except Exception as e:
            st.error(f"Error initializing Cohere client: {e}")
            st.stop()

        try:
            self.compass_client = CompassClient(
                index_url=compass_url,
                bearer_token=compass_token
            )
        except Exception as e:
            st.error(f"Error initializing Compass client: {e}")
            st.stop()

        self.research_steps = []
        self.collected_evidence = []
        self.confidence_threshold = CONFIDENCE_THRESHOLD

    # Hybrid search + Rerank
    def get_relevant_chunks(self, query: str, top_k: int) -> List[Dict]:
        """
        Search the 'compass-rev' index using Compass. Return a list of chunk dicts.
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
                    # Defensive checks
                    if not hasattr(hit, "content") or hit.content is None:
                        st.warning(f"Warning: 'hit.content' is missing for hit index {idx}")
                        continue
                    if not isinstance(hit.content, dict):
                        st.warning(f"Warning: 'hit.content' is not a dict for hit index {idx}, got {type(hit.content)}")
                        continue

                    text = hit.content.get("text", "")
                    # If 'score' is missing, default to 1.0
                    score = getattr(hit, "score", 1.0)

                    # Filter out low-score hits
                    if score >= self.confidence_threshold:
                        documents.append({
                            "title": f"doc_{idx}",
                            "snippet": text,
                            "score": score
                        })
            else:
                st.info("No hits returned from Compass for query: " + query)
            return documents

        except Exception as e:
            st.error(f"Error retrieving documents from Compass: {e}")
            return []

    # Query Decomposition & Expansion
    def plan_research(self, user_query: str) -> List[str]:
        """
        Use LLM to generate possible expansions/sub-queries.
        We'll do a short prompt to get ~2-5 sub-queries.
        """
        system_message = (
            "You are a helpful assistant that breaks down complex questions into 2-5 smaller search queries. "
            "These queries should collectively gather all relevant information from transcripts. "
            "If the user query is already simple, just return it unchanged."
        )
        prompt = (
            f"{system_message}\n\n"
            f"User Query: {user_query}\n\n"
            "Provide 2-5 bullet points, each a short search query or sub-question."
        )
        try:
            response = self.co.chat(
                message=prompt,
                model="command-r-plus-08-2024",
                temperature=0.2
            )
        except Exception as e:
            st.error(f"Error calling Cohere for plan_research: {e}")
            return [user_query]  # fallback

        lines = response.text.strip().split("\n")
        steps = []
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and (line_stripped[0].isdigit() or line_stripped.startswith("-")):
                cleaned = line_stripped.lstrip("0123456789.-) ")
                steps.append(cleaned)

        if not steps:
            steps = [user_query]  # fallback

        return steps[:5]

    # Check if we have enough evidence
    def analyze_evidence(self, query: str, evidence: List[Dict]) -> Dict:
        """
        Ask LLM if the info is sufficient. Return a dict with is_complete/gaps/next_query.
        """
        evidence_text = ""
        for i, e in enumerate(evidence):
            snippet = e["snippet"][:500]
            evidence_text += f"Snippet {i+1} (score={e['score']}):\n{snippet}\n\n"

        sys_prompt = (
            "You are a world-class analyst. You only rely on the provided transcript excerpts "
            "and do NOT hallucinate. If not enough info is present, you must say so."
        )
        prompt = (
            f"{sys_prompt}\n\n"
            f"User query: '{query}'\n\n"
            f"Relevant transcript excerpts:\n{evidence_text}\n\n"
            "Based on these excerpts, do we have enough info to provide a detailed, correct answer? "
            "If not, suggest what is missing or a new search query.\n\n"
            "Respond in valid JSON:\n"
            "{\n"
            "  \"is_complete\": true/false,\n"
            "  \"gaps\": [\"gap1\", \"gap2\"],\n"
            "  \"next_query\": \"some new search query or null\"\n"
            "}\n"
        )
        try:
            response = self.co.chat(
                message=prompt,
                model="command-r-plus-08-2024",
                temperature=0.2
            )
        except Exception as e:
            st.error(f"Error calling Cohere for analyze_evidence: {e}")
            # Default to "we have enough" to avoid infinite loop
            return {
                "is_complete": True,
                "gaps": [],
                "next_query": None
            }

        raw_text = response.text.strip()
        # Debug print
        st.write("DEBUG - analyze_evidence raw LLM response:", raw_text)

        try:
            parsed = json.loads(raw_text)
            if not isinstance(parsed, dict):
                raise ValueError("Cohere returned data but it's not a JSON object.")
            if "is_complete" not in parsed:
                parsed["is_complete"] = True
            if "gaps" not in parsed:
                parsed["gaps"] = []
            if "next_query" not in parsed:
                parsed["next_query"] = None
            return parsed
        except Exception as e:
            st.warning(f"Unable to parse JSON from analyze_evidence response: {e}")
            st.write("DEBUG - Fallback to is_complete = True")
            return {
                "is_complete": True,
                "gaps": [],
                "next_query": None
            }

    # Synthesize final answer
    def synthesize_answer(self, query: str, evidence: List[Dict]) -> str:
        """
        Generate a final coherent answer from the transcript snippets only.
        """
        if not evidence:
            return "I do not have enough information from the transcripts."

        evidence_text = ""
        for i, e in enumerate(evidence):
            snippet = e["snippet"].replace("\n", " ")
            evidence_text += f"[Source doc_{i}, score={e['score']:.2f}]: {snippet}\n\n"

        sys_prompt = (
            "You are a careful, fact-based assistant that ONLY uses the provided transcript snippets to answer. "
            "If the information is not in the snippets, say 'I do not have enough information from the transcripts'."
        )
        prompt = f"""
{sys_prompt}

User asked:
"{query}"

Below are relevant transcript excerpts:
{evidence_text}

Task:
Provide a thorough, cohesive answer to the user's query, referencing [Source doc_i] whenever pulling specifics.
If there's insufficient info, say: "I do not have enough information from the transcripts."

Final Answer:
"""
        try:
            response = self.co.chat(
                message=prompt,
                model="command-r-plus-08-2024",
                temperature=0.0
            )
            return response.text.strip()
        except Exception as e:
            st.error(f"Error calling Cohere for synthesize_answer: {e}")
            return "I'm sorry, I encountered an error generating the answer."

    # Multi-step Orchestrator
    def research(self, user_query: str, max_rounds: int, top_k: int) -> Dict:
        """
        1) plan_research -> subqueries
        2) for each subquery, retrieve
        3) accumulate, check if complete
        """
        self.research_steps = []
        self.collected_evidence = []

        # Plan
        planned_queries = self.plan_research(user_query)
        round_count = 0

        for sq in planned_queries:
            if round_count >= max_rounds:
                break

            self.research_steps.append({"step": round_count+1, "sub_query": sq})
            new_evidence = self.get_relevant_chunks(sq, top_k=top_k)
            if new_evidence:
                self.collected_evidence.extend(new_evidence)
            else:
                st.info(f"No new evidence retrieved for sub-query: {sq}")

            analysis = self.analyze_evidence(user_query, self.collected_evidence)
            round_count += 1

            if analysis["is_complete"]:
                break
            else:
                if analysis["next_query"]:
                    st.write(f"DEBUG - Next query suggested: {analysis['next_query']}")
                    planned_queries.append(analysis["next_query"])

        final_answer = self.synthesize_answer(user_query, self.collected_evidence)
        return {
            "query": user_query,
            "steps": self.research_steps,
            "evidence": self.collected_evidence,
            "answer": final_answer
        }

# -----------------------------------------
# Streamlit UI (Simple)
# -----------------------------------------
st.set_page_config(
    page_title="Transcript RAG - Simple Demo",
    page_icon="🔍",
    layout="wide"
)

st.title("Transcript Q&A (Simple) - Debug Version")

# If not in session, init agent
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

# Prompt
query = st.text_area("Enter your question about the transcripts:", height=100)

# Single button
if st.button("Submit Question"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching transcripts..."):
            try:
                result = st.session_state.agent.research(
                    user_query=query,
                    max_rounds=MAX_ROUNDS,
                    top_k=TOP_K
                )
                answer = result["answer"]
            except Exception as e:
                st.error("An unexpected error occurred in `research`:")
                st.exception(e)
                st.stop()

        st.subheader("Answer")
        st.write(answer)

        # Show debug details about retrieval steps:
        with st.expander("Show Retrieval Evidence (Debug)"):
            st.markdown("#### Research Steps")
            for step_data in result["steps"]:
                st.write(f"**Step {step_data['step']}**: {step_data['sub_query']}")

            st.markdown("#### Evidence Chunks")
            for idx, ev in enumerate(result["evidence"]):
                st.write(f"**Source doc_{idx}** (score: {ev['score']:.2f})")
                st.write(ev["snippet"])
                st.markdown("---")
