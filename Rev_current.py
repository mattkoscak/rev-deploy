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
MAX_RESEARCH_STEPS = 5      # how many times we let the model "search" before forcing an answer
RETRIEVAL_TOP_K = 5         # how many chunks we retrieve each search
TEMPERATURE = 0.0           # keep it low for factual tasks
COMMAND_MODEL = "command-nightly"  # example cohere model; adapt as needed

# -----------------------------------------
# TranscriptRAGAgent Class
# -----------------------------------------
class TranscriptRAGAgent:
    """
    A 'Deep Research'-style agent using Cohere + Compass.
    It iteratively decides whether it has enough info to answer.
    If not, it requests a 'search' action, we retrieve with Compass,
    then feed the results back in. Finally it yields an 'answer' action.
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
        Return them as a list of snippet strings (and maybe store metadata).
        """
        try:
            results = self.compass_client.search_chunks(index_name="compass-rev", query=query, top_k=top_k)
            snippets = []
            if results.hits:
                for idx, h in enumerate(results.hits):
                    text = h.content.get("text", "")
                    score = getattr(h, "score", 1.0)
                    # You can store the ID or doc metadata too, e.g. h.id
                    # For simplicity, just keep snippet text:
                    snippet_label = f"[doc_{idx}, score={score:.2f}]: {text}"
                    snippets.append(snippet_label)
            return snippets
        except Exception as e:
            st.warning(f"Compass retrieval error: {e}")
            return []

    def run_deep_research(self, user_query: str, style: str = "default") -> Dict:
        """
        Core loop: keep asking the model if it wants to search or answer.
        We store each message in a conversation list. If model says 'search',
        we retrieve from Compass, append those results as context, and ask again.
        If model says 'answer', we finalize.
        """

        # We keep track of the entire conversation as (role, content) pairs
        # plus an array of "debug steps" if you want to display them.
        conversation = []
        debug_steps = []

        # 1) System instructions for the model:
        system_msg = (
            "You are a helpful 'Deep Research' AI. You can do two actions:\n"
            "1) \"search\": if you need more info, output JSON like {\"action\": \"search\", \"query\": \"...\"}\n"
            "2) \"answer\": if you have enough info, output JSON like {\"action\": \"answer\", \"content\": \"...\"}\n\n"
            "Rules:\n"
            "- Provide references in your final answer if they come from search snippets (e.g. [doc_0]).\n"
            "- Do NOT fabricate references.\n"
            "- If the user’s question cannot be answered from the retrieved snippets, say so.\n"
            "- Keep your chain-of-thought hidden. Only output valid JSON.\n"
            f"- Style: {style} (concise, verbose, or default)."
        )
        conversation.append({"role": "system", "content": system_msg})
        conversation.append({"role": "user", "content": user_query})

        # 2) Iterative loop
        steps_taken = 0
        final_answer = "No answer generated."
        while steps_taken < MAX_RESEARCH_STEPS:
            # Prepare the text to send to Cohere
            all_messages_text = self.build_faux_chat_prompt(conversation)

            # Call Cohere to get the next JSON action
            response = self.co.generate(
                model=COMMAND_MODEL,
                prompt=all_messages_text,
                max_tokens=300,
                temperature=TEMPERATURE,
                stop_sequences=["\n{\"action\""],
            )
            assistant_text = response.generations[0].text.strip()

            # Store debug
            debug_steps.append(f"LLM Output:\n{assistant_text}")

            # Attempt to parse JSON
            parsed_action = self.safe_json_parse(assistant_text)

            if not parsed_action:
                # If parsing fails, we’ll just end
                final_answer = (
                    "I'm sorry, I couldn't parse the model's response. "
                    "Something went wrong in the chain of thought."
                )
                break

            action = parsed_action.get("action", None)
            if action == "search":
                # The model wants to search
                query_for_docs = parsed_action.get("query", "")
                if not query_for_docs.strip():
                    # no valid query, so break
                    final_answer = "The model requested a search, but no valid query was provided."
                    break

                # Retrieve from Compass
                retrieved_snippets = self.retrieve_docs(query_for_docs, top_k=RETRIEVAL_TOP_K)

                if len(retrieved_snippets) == 0:
                    # No results found, we can either break or let the model try again
                    conversation.append({
                        "role": "system",
                        "content": "No documents were found for that search query."
                    })
                else:
                    # Append as a system message describing the results
                    # We limit how many chars to avoid prompt overload
                    snippet_text_block = "\n\n".join(retrieved_snippets)
                    if len(snippet_text_block) > 6000:
                        snippet_text_block = snippet_text_block[:6000] + "... (truncated)"

                    conversation.append({
                        "role": "system",
                        "content": f"Search results for '{query_for_docs}':\n{snippet_text_block}"
                    })

            elif action == "answer":
                # The model is ready to provide a final answer
                final_answer = parsed_action.get("content", "No content.")
                break
            else:
                # Unrecognized action
                final_answer = f"Unrecognized action: {action}"
                break

            steps_taken += 1

        # If we exit loop without an "answer" action, we can forcibly produce one
        if steps_taken == MAX_RESEARCH_STEPS and action != "answer":
            # Let the model produce a final answer with the current context
            # You might forcibly tell the model: “Please finalize an answer now.”
            conversation.append({
                "role": "system",
                "content": "You have reached the max steps; please provide your final answer now using {\"action\":\"answer\",\"content\":\"...\"}."
            })
            all_messages_text = self.build_faux_chat_prompt(conversation)
            response = self.co.generate(
                model=COMMAND_MODEL,
                prompt=all_messages_text,
                max_tokens=400,
                temperature=TEMPERATURE,
                stop_sequences=["\n{\"action\""],
            )
            assistant_text = response.generations[0].text.strip()
            parsed_action = self.safe_json_parse(assistant_text)
            if parsed_action and parsed_action.get("action") == "answer":
                final_answer = parsed_action["content"]

        return {
            "query": user_query,
            "answer": final_answer,
            "debug_steps": debug_steps
        }

    def build_faux_chat_prompt(self, conversation: List[Dict]) -> str:
        """
        Cohere doesn't have a built-in multi-message chat by default
        (unless using the Chat endpoint), so we can manually build a 'chat-like' prompt.
        Each role is prefixed ("System:", "User:", "Assistant:") to emulate conversation.
        """
        prompt_lines = []
        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            if role == "system":
                prompt_lines.append(f"System: {content}\n")
            elif role == "user":
                prompt_lines.append(f"User: {content}\n")
            else:
                prompt_lines.append(f"Assistant: {content}\n")
        # We finish with "Assistant:" so the generation picks up from the AI's perspective
        prompt_lines.append("Assistant: ")
        return "".join(prompt_lines)

    def safe_json_parse(self, text: str) -> Optional[Dict]:
        """
        Try to parse the model's response as JSON. If it fails, return None.
        We assume the model’s entire response is a JSON object
        like { "action": "...", "query": "..."} or { "action": "answer", "content": "..."}.
        """
        try:
            return json.loads(text)
        except:
            return None

# -----------------------------------------
# Streamlit UI
# -----------------------------------------
st.set_page_config(
    page_title="Transcript Q&A - Deep Research style",
    layout="wide"
)

st.title("Deep Research-Style RAG Chatbot")

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
        with st.spinner("Researching..."):
            result_dict = st.session_state.agent.run_deep_research(query, style_mode)

        # Display final answer
        st.subheader("Answer")
        st.write(result_dict["answer"])

        # Debug info
        with st.expander("Show Debug Steps"):
            for i, step in enumerate(result_dict["debug_steps"]):
                st.markdown(f"**Step {i+1}**\n```\n{step}\n```")
