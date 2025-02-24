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
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input("Password", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("Incorrect password.")
    return st.session_state.get("password_correct", False)

if not check_password():
    st.stop()

# -----------------------------------------
# Hidden config
# -----------------------------------------
SINGLE_RETRIEVAL_TOP_K = 10
MULTI_RETRIEVAL_TOP_K = 8
MAX_ROUNDS = 3
CONFIDENCE_THRESHOLD = 0.0

# -----------------------------------------
# TranscriptRAGAgent Class
# -----------------------------------------
class TranscriptRAGAgent:
    """
    Incorporates:
    - Heuristic to decide single-hop vs multi-hop
    - If multi-hop, do the plan_research approach
    - Mandatory references in final answer
    - Output styles (concise, default, verbose)
    """

    def __init__(self, compass_url: str, compass_token: str, cohere_api_key: str):
        try:
            self.co = CohereClient(api_key=cohere_api_key, client_name="transcript-rag-agent")
        except Exception as e:
            st.error(f"Error initializing Cohere client: {e}")
            st.stop()
        try:
            self.compass_client = CompassClient(index_url=compass_url, bearer_token=compass_token)
        except Exception as e:
            st.error(f"Error initializing Compass client: {e}")
            st.stop()

        self.confidence_threshold = CONFIDENCE_THRESHOLD

    # --- Heuristic: LLM call to decide multi-hop or single-hop
    def decide_if_multihop(self, user_query: str) -> bool:
        """
        A quick LLM-based classification:
        Return True if multi-hop needed, False if single-hop likely suffices.
        """
        system_prompt = (
            "You are a classification model. The user question might need multiple sub-queries (multi-hop) or not.\n"
            "Please return valid JSON with a single boolean field 'is_multihop'."
        )
        decision_prompt = (
            f"{system_prompt}\n\nUser question: '{user_query}'\n"
            "Decide if multi-hop is needed. For example, if the question is complex, has multiple parts, or requires combining info across different transcripts.\n\n"
            "Output JSON:\n{\n  \"is_multihop\": true or false\n}"
        )
        try:
            resp = self.co.chat(message=decision_prompt, model="command-r-plus-08-2024", temperature=0.2)
            raw_text = resp.text.strip()
            # Attempt parse
            parsed = json.loads(raw_text)
            if "is_multihop" in parsed and isinstance(parsed["is_multihop"], bool):
                return parsed["is_multihop"]
            # fallback
            return False
        except Exception as e:
            st.warning(f"Heuristic LLM error, defaulting single-hop: {e}")
            return False

    # --- Single-hop retrieval
    def retrieve_once(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Simple direct retrieval from Compass, return chunk list.
        """
        try:
            results = self.compass_client.search_chunks(index_name="compass-rev", query=query, top_k=top_k)
            docs = []
            if results.hits:
                for idx, h in enumerate(results.hits):
                    text = h.content.get("text", "")
                    score = getattr(h, "score", 1.0)
                    if score >= self.confidence_threshold:
                        docs.append({
                            "title": f"doc_{idx}",
                            "snippet": text,
                            "score": score
                        })
            return docs
        except Exception as e:
            st.error(f"Compass error: {e}")
            return []

    # --- Multi-hop approach
    def plan_research(self, user_query: str) -> List[str]:
        """
        LLM suggests sub-queries if the question is complex.
        """
        sys_msg = (
            "You are a helpful assistant that breaks down the question into 2-5 sub-queries if needed, otherwise return it unchanged."
        )
        prompt = (
            f"{sys_msg}\n\nUser Query: {user_query}\n\n"
            "Provide 2-5 bullet points, each sub-query or short search query."
        )
        try:
            resp = self.co.chat(message=prompt, model="command-r-plus-08-2024", temperature=0.2)
            lines = resp.text.strip().split("\n")
            steps = []
            for line in lines:
                line_stripped = line.strip()
                if line_stripped and (line_stripped[0].isdigit() or line_stripped.startswith("-")):
                    cleaned = line_stripped.lstrip("0123456789.-) ")
                    steps.append(cleaned)
            if not steps:
                return [user_query]
            return steps[:5]
        except:
            return [user_query]

    def analyze_evidence(self, query: str, evidence: List[Dict]) -> Dict:
        """
        Check if we have enough info. Return a JSON with is_complete, next_query, etc.
        """
        snippet_text = ""
        for i, e in enumerate(evidence):
            snippet_text += f"[doc_{i}, score={e['score']:.2f}]: {e['snippet'][:300]}\n\n"

        sys_msg = (
            "You are a thorough analyst. Only rely on the provided text. If more info is needed, propose a new sub-query."
        )
        prompt = (
            f"{sys_msg}\n\nUser query: {query}\n\nSnippets:\n{snippet_text}\n\n"
            "Are these sufficient? If not, propose next_query.\n"
            "Respond in JSON:\n"
            "{ \"is_complete\": true/false, \"next_query\": \"some text or null\" }"
        )
        try:
            resp = self.co.chat(message=prompt, model="command-r-plus-08-2024", temperature=0.2)
            raw = resp.text.strip()
            parsed = json.loads(raw)
            if "is_complete" not in parsed:
                parsed["is_complete"] = True
            if "next_query" not in parsed:
                parsed["next_query"] = None
            return parsed
        except:
            return {"is_complete": True, "next_query": None}

    # -- Multi-step orchestrator
    def multi_hop_research(self, user_query: str, max_rounds: int = 3, top_k: int = 8) -> List[Dict]:
        """
        Plan sub-queries => retrieve => accumulate => check if complete => possibly new query => done
        Return the final list of combined evidence
        """
        final_evidence = []
        sub_queries = self.plan_research(user_query)
        round_count = 0
        for sq in sub_queries:
            if round_count >= max_rounds:
                break
            new_evs = self.retrieve_once(sq, top_k)
            final_evidence.extend(new_evs)
            analysis = self.analyze_evidence(user_query, final_evidence)
            if analysis["is_complete"]:
                break
            if analysis["next_query"]:
                sub_queries.append(analysis["next_query"])
            round_count += 1
        return final_evidence

    # -- Single method to gather context
    def gather_evidence(self, user_query: str) -> List[Dict]:
        # 1) Decide if multi-hop
        is_multi = self.decide_if_multihop(user_query)
        if is_multi:
            st.write("DEBUG: Multi-hop approach triggered.")
            return self.multi_hop_research(user_query, MAX_ROUNDS, MULTI_RETRIEVAL_TOP_K)
        else:
            st.write("DEBUG: Single-hop approach triggered.")
            return self.retrieve_once(user_query, SINGLE_RETRIEVAL_TOP_K)

    # -- Synthesis with mandatory snippet references + style
    def synthesize_answer(self, query: str, evidence: List[Dict], style: str = "default") -> str:
        if not evidence:
            return "I do not have enough information from the transcripts."

        # Merge snippet text
        snippet_block = ""
        for i, e in enumerate(evidence):
            # keep it shorter if there's a lot
            snippet_block += f"[doc_{i}, score={e['score']:.2f}]: {e['snippet']}\n\n"

        # Style instructions
        if style == "concise":
            style_instructions = (
                "Respond in a short, concise bullet-point style. "
                "For each bullet, reference at least one [doc_i]. "
                "Omit any bullet that lacks snippet support."
            )
        elif style == "verbose":
            style_instructions = (
                "Respond in a thorough, detailed format. "
                "Use paragraphs and more context from the snippets. "
                "Still, each major statement must cite [doc_i]. "
                "If no snippet supports a statement, omit it."
            )
        else:
            style_instructions = (
                "Provide a clear, structured bullet-point answer. "
                "Each bullet must cite [doc_i]."
            )

        sys_prompt = (
            "You are a careful, fact-based assistant that ONLY uses the provided transcript snippets. "
            "If the information is not there, say so. "
            "Do NOT add any knowledge not found in the snippets."
        )
        final_prompt = f"""
{sys_prompt}

User asked: "{query}"

Snippets:
{snippet_block}

Style Guidance: {style_instructions}

Final Answer:
"""
        try:
            resp = self.co.chat(message=final_prompt, model="command-r-plus-08-2024", temperature=0.0)
            return resp.text.strip()
        except Exception as e:
            st.error(f"Error in synthesis LLM call: {e}")
            return "I'm sorry, I encountered an error generating the final answer."

    # -- Full pipeline
    def answer_query(self, user_query: str, style: str) -> Dict:
        evidence = self.gather_evidence(user_query)
        final_answer = self.synthesize_answer(user_query, evidence, style)
        return {
            "query": user_query,
            "evidence": evidence,
            "answer": final_answer
        }

# -----------------------------------------
# Streamlit UI
# -----------------------------------------
st.set_page_config(
    page_title="Transcript RAG - Multi/Single-hop + Mandatory Citations",
    layout="wide"
)

st.title("Transcript Q&A - Multi vs Single-hop with 3 Styles")

# Initialize agent
if 'agent' not in st.session_state:
    required_vars = ["COHERE_API_KEY", "COMPASS_TOKEN", "COMPASS_URL"]
    missing_vars = [v for v in required_vars if not os.environ.get(v)]
    if missing_vars:
        st.error(f"Missing environment variables: {', '.join(missing_vars)}")
        st.stop()
    st.session_state.agent = TranscriptRAGAgent(
        compass_url=os.environ["COMPASS_URL"],
        compass_token=os.environ["COMPASS_TOKEN"],
        cohere_api_key=os.environ["COHERE_API_KEY"]
    )

query = st.text_area("Enter your question about the transcripts:", height=100)
style_mode = st.selectbox(
    "Answer Style",
    options=["default", "concise", "verbose"],
    index=0,
    help="Choose the level of detail for the final answer."
)

if st.button("Submit Question"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Gathering information..."):
            result_dict = st.session_state.agent.answer_query(query, style_mode)
            final_answer = result_dict["answer"]

        st.subheader("Answer")
        st.write(final_answer)

        with st.expander("Show Evidence & Debug"):
            st.markdown("**Retrieved Evidence**")
            for idx, e in enumerate(result_dict["evidence"]):
                st.write(f"doc_{idx} (score={e['score']:.2f})")
                st.write(e["snippet"][:300] + "...")
                st.markdown("---")
