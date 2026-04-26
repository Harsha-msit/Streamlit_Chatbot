import streamlit as st
from rag_pipeline import RAGPipeline

st.title("📚 RAG AI Assistant")

st.markdown("Ask questions based on your documents 📄")

@st.cache_resource
def load_pipeline():
    return RAGPipeline()

rag = load_pipeline()

if "history" not in st.session_state:
    st.session_state.history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

def build_context(messages, k=5):
    recent = messages[-k:]
    context = ""

    for role, text in recent:
        context += f"{role}: {text}\n"

    return context

query = st.text_input("💬 Your question:")
st.write("AI Response...")

if query:
    # st.session_state.history.append(("You", query))
    st.session_state.messages.append(("User", query))

    context = build_context(st.session_state.messages)

    response_placeholder = st.empty()
    full_text = ""

    for chunk in rag.stream(context):
        full_text += chunk
        response_placeholder.markdown(full_text)

    # st.session_state.history.append(("AI", full_text))
    st.session_state.messages.append(("AI", full_text))

st.write("Conversation History:")
for role, text in st.session_state.messages:
    if role == "User":
        st.markdown(f"🧑 **You:** {text}")
    else:
        st.markdown(f"🤖 **AI:** {text}")

# # Show history
# st.write("Conversation History:")
# for role, text in st.session_state.history:
#     st.write(f"**{role}:** {text}")