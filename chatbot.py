import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import create_engine, inspect, text

# Load environment variables
load_dotenv()

# Initialize Streamlit
st.set_page_config(page_title="💬 Course Chatbot", page_icon="💬", layout="wide")
st.title("💬 Course Chatbot")
st.markdown("Ask anything about your course! Chats are saved permanently.")

# --- Database Setup ---
DB_URL = "sqlite:///chat_history.db"
engine = create_engine(DB_URL)

# Initialize database schema if needed
def initialize_database():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS message_store (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message TEXT NOT NULL,
                type TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()

initialize_database()

# --- Database Management Functions ---
def inspect_database():
    inspector = inspect(engine)
    
    if not inspector.get_table_names():
        st.warning("No tables found in the database.")
        return
    
    st.subheader("Database Schema Inspection")
    
    for table_name in inspector.get_table_names():
        st.write(f"Table: {table_name}")
        columns = inspector.get_columns(table_name)
        for column in columns:
            st.write(f"- {column['name']}: {column['type']}")

def delete_database_file():
    db_file = "chat_history.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        st.success("Database file deleted successfully. A new one will be created automatically.")
        initialize_database()  # Reinitialize the database
    else:
        st.warning("Database file not found.")

# --- Session Management ---
def initialize_session():
    if "session_counter" not in st.session_state:
        st.session_state.session_counter = 0
    if "current_session" not in st.session_state:
        st.session_state.current_session = f"user_{st.session_state.session_counter}"
        st.session_state.session_counter += 1
    if "messages" not in st.session_state:
        st.session_state.messages = []

# Get all available sessions from database
def get_all_sessions():
    inspector = inspect(engine)
    if "message_store" not in inspector.get_table_names():
        return []
    
    with engine.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT session_id FROM message_store"))
        return [row[0] for row in result.fetchall()]

# Initialize chat history with SQL backend
@st.cache_resource
def get_chat_history(session_id):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=engine
    )

# --- New Chat Function ---
def start_new_chat():
    st.session_state.current_session = f"user_{st.session_state.session_counter}"
    st.session_state.session_counter += 1
    st.session_state.messages = []

# --- UI Components ---
initialize_session()

# Sidebar for conversation history
with st.sidebar:
    st.header("Conversation History")
    
    # Button for new chat
    if st.button("🔄 Start New Chat"):
        start_new_chat()
        st.rerun()
    
    st.markdown("---")
    st.subheader("Previous Conversations")
    
    # Get all sessions and display them
    all_sessions = get_all_sessions()
    current_session = st.session_state.current_session
    
    # Display sessions with most recent first
    for session in reversed(all_sessions):
        # Get the first message as a title (or use session ID)
        try:
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT message FROM message_store WHERE session_id = :session_id LIMIT 1"),
                    {"session_id": session}
                )
                first_message = result.fetchone()
        except Exception as e:
            st.warning(f"Error loading session {session}: {str(e)}")
            continue
        
        session_title = first_message[0][:30] + "..." if first_message else f"Chat {session}"
        
        # Display session with radio button for selection
        if st.button(
            f"🗨️ {session_title}",
            key=f"session_{session}",
            use_container_width=True,
            type="primary" if session == current_session else "secondary"
        ):
            st.session_state.current_session = session
            st.rerun()
    
    # Database management section
    st.markdown("---")
    st.subheader("Database Management")
    if st.button("🔍 Inspect Database Schema"):
        inspect_database()
    
    if st.button("🗑️ Reset Database (Delete chat_history.db)"):
        delete_database_file()
        st.rerun()

# --- LLM Setup ---
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("Set OPENROUTER_API_KEY in .env")
    st.stop()

llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=api_key,
    model_name="deepseek/deepseek-r1-0528-qwen3-8b:free",
    temperature=0.7,
    default_headers={
        "HTTP-Referer": "https://your-streamlit-app-url",
        "X-Title": "CourseChatBot"
    }
)

# --- Chat Chain ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful course assistant. Be detailed and friendly."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: get_chat_history(session_id),
    input_messages_key="input",
    history_messages_key="history",
)

# --- Main Chat Interface ---
current_session = st.session_state.current_session
msgs = get_chat_history(current_session)

# Display current conversation
st.subheader(f"Current Conversation: {current_session}")

# Display chat messages from history
for message in msgs.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# Handle new input
if prompt_input := st.chat_input("Ask about your course..."):
    # Display user message
    with st.chat_message("human"):
        st.markdown(prompt_input)
    
    # Add to chat history
    msgs.add_user_message(prompt_input)
    
    try:
        with st.spinner("Thinking..."):
            # Get and display response
            response = chain_with_history.invoke(
                {"input": prompt_input},
                config={"configurable": {"session_id": current_session}}
            )
            
            with st.chat_message("ai"):
                st.markdown(response.content)
            
            msgs.add_ai_message(response.content)
    except Exception as e:
        st.error(f"Error: {str(e)}")