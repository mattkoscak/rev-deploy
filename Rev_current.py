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
# Configurable Parameters
# -----------------------------------------
MAX_RESEARCH_STEPS = 5      # Maximum times we let the model do "search" before forcing an answer
RETRIEVAL_TOP_K = 5         # Number of chunks from Compass on each search
TEMPERATURE = 0.0           # Keep it low for factual tasks
COMMAND_MODEL = "command-nightly"  # Example Cohere model name

# -----------------------------------------
# TranscriptRAGAgent Class
# -----------------------------------------
class TranscriptRAGAgent:
    """
    A 'Deep Research'-style agent using Cohere + Compass.
    It iteratively decides whether it has enough info to answer,
    but we REQUIRE at least one retrieval step before finalizing.
    """

    def __init__(self, compass_url: str, compass_token: str, cohere_api_key: str):
        # Initialize Cohere client
        try:
            self.co = CohereClient(api_key=cohere_api_key)
        except Exception as e:
            st.error(f"Error initializing Cohere client: {e}")
            st.stop()

        # Initialize Compass client
        try:
            self.compass_client = CompassClient(index_url=compass_url, bearer_token=compass_token)
        except Exception as e:
            st.error(f"Error initializing Compass client: {e}")
            st.stop()

    def retrieve_docs(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve top_k chunks from Compass for a given query.
        Return them as a list of snippet strings (with doc_x labels).
        """
        try:
            results = self.compass_client.search_chunks(index_name="compass-rev", query=query, top_k=top_k)
            snippets = []
            if results.hits:
                for idx, h in enumerate(results.hits):
                    text = h.content.get("text", "")
                    score = getattr(h, "score", 1.0)
                    snippet_label = f"[doc_{idx}, score={score:.2f}]: {text}"
                    snippets.append(snippet_label)
            return snippets
        except Exception as e:
            st.warning(f"Compass retrieval error: {e}")
            return []

    def run_deep_research(self, user_query: str, style: str = "default") -> Dict:
        """
        Core loop: keep asking the model if it wants to search or answer.
        We require at least one 'search' before it can produce a final 'answer'.
        """

        conversation = []
        debug_steps = []

        # 1) System instructions
        system_msg = (
            "You are a helpful 'Deep Research' AI. You have two possible actions:\n"
            "1) \"search\": If you need more info, return JSON: {\"action\": \"search\", \"query\": \"...\"}\n"
            "2) \"answer\": If you have enough info, return JSON: {\"action\": \"answer\", \"content\": \"...\"}\n"
            "But you MUST perform at least one 'search' action before giving the final answer.\n\n"
            "Additional rules:\n"
            "- You may do multiple searches if the first results were insufficient.\n"
            "- Provide references in your final answer if they come from retrieved snippets (e.g. [doc_0]).\n"
            "- Do not fabricate references.\n"
            "- If no data is found or it doesn't exist, you may finalize with an appropriate statement.\n"
            "- Keep chain-of-thought hidden; only output valid JSON.\n"
            f"- Style: {style}\n"
        )
        conversation.append({"role": "system", "content": system_msg})
        conversation.append({"role": "user", "content": user_query})

        steps_taken = 0
        searches_done = 0
        final_answer = "No answer produced."

        while steps_taken < MAX_RESEARCH_STEPS:
            # Build the prompt from conversation so far
            prompt_text = self.build_faux_chat_prompt(conversation)

            # Call Cohere to get next action
            response = self.co.generate(
                model=COMMAND_MODEL,
                prompt=prompt_text,
                max_tokens=300,
                temperature=TEMPERATURE,
                stop_sequences=["\n{\"action\""],
            )
            assistant_text = response.generations[0].text.strip()
            debug_steps.append(f"LLM Output:\n{assistant_text}")

            # Parse JSON
            parsed = self.safe_json_parse(assistant_text)
            if not parsed:
                # If we can't parse, break
                final_answer = "Could not parse the model's response as valid JSON."
                break

            action = parsed.get("action", "")
            if action == "search":
                # The model wants to search
                search_query = parsed.get("query", "").strip()
                if not search_query:
                    final_answer = "Model requested a search but gave no query. Ending."
                    break

                # Actually do the retrieval
                results = self.retrieve_docs(search_query, top_k=RETRIEVAL_TOP_K)

                if len(results) == 0:
                    # No results found
                    conversation.append({
                        "role": "system",
                        "content": f"No documents found for search query: '{search_query}'."
                    })
                else:
                    # Add results as a system message
                    snippet_block = "\n\n".join(results)
                    if len(snippet_block) > 6000:
                        snippet_block = snippet_block[:6000] + "... (truncated)"
                    conversation.append({
                        "role": "system",
                        "content": f"Search results for '{search_query}':\n{snippet_block}"
                    })

                searches_done += 1

            elif action == "answer":
                # The model is trying to produce a final answer
                if searches_done == 0:
                    # Force at least one search
                    conversation.append({
                        "role": "system",
                        "content": (
                            "You have not performed any search yet. You MUST do at least one search before finalizing."
                        )
                    })
                    # We do NOT break; we let the loop continue so the model can try again
                else:
                    # Accept the final answer
                    final_answer = parsed.get("content", "")
                    break
            else:
                final_answer = f"Unrecognized action: {action}. Exiting."
                break

            steps_taken += 1

        # If we exit due to steps limit
        if steps_taken == MAX_RESEARCH_STEPS and searches_done > 0 and action != "answer":
            # Force the model to finalize now
            conversation.append({
                "role": "system",
                "content": "You have reached max steps. Produce a final answer now, using {\"action\":\"answer\",\"content\":\"...\"}."
            })
            prompt_text = self.build_faux_chat_prompt(conversation)
            response = self.co.generate(
                model=COMMAND_MODEL,
                prompt=prompt_text,
                max_tokens=400,
                temperature=TEMPERATURE,
                stop_sequences=["\n{\"action\""],
            )
            assistant_text = response.generations[0].text.strip()
            debug_steps.append(f"LLM Output (forced final):\n{assistant_text}")

            parsed = self.safe_json_parse(assistant_text)
            if parsed and parsed.get("action") == "answer":
                final_answer = parsed.get("content", "No content.")
            else:
                final_answer = "No valid final answer returned."

        return {
            "query": user_query,
            "answer": final_answer,
            "debug_steps": debug_steps
        }

    def build_faux_chat_prompt(self, conversation: List[Dict]) -> str:
        """
        Build a 'chat-like' prompt for Cohere from the conversation list.
        """
        lines = []
        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            if role == "system":
                lines.append(f"System: {content}\n")
            elif role == "user":
                lines.append(f"User: {content}\n")
            else:
                lines.append(f"Assistant: {content}\n")
        # Prompt ends with "Assistant: "
        lines.append("Assistant: ")
        return "".join(lines)

    def safe_json_parse(self, text: str) -> Optional[Dict]:
        """
        Try to parse the model's response as JSON. If fail, return None.
        """
        try:
            return json.loads(text)
        except:
            return None

# -----------------------------------------
# Streamlit UI
# -----------------------------------------
st.set_page_config(
    page_title="Deep Research-Style RAG Chatbot (Forced Search)",
    layout="wide"
)

st.title("Deep Research-Style RAG Chatbot (with Forced Minimum 1 Search)")

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
        with st.spinner("Researching..."):
            result_dict = st.session_state.agent.run_deep_research(query, style_mode)

        st.subheader("Answer")
        st.write(result_dict["answer"])

        with st.expander("Show Debug Steps"):
            for i, step in enumerate(result_dict["debug_steps"]):
                st.markdown(f"**Step {i+1}**\n```\n{step}\n```")
