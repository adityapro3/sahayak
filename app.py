import streamlit as st
import ollama
import os
import sqlite3
import json
import re
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import ChatCohere
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Sahayak: Your Learning Assistant",
    page_icon="📚",
    layout="wide",
)

# --- API Key Configuration ---
# Securely manage API keys using Streamlit's secrets.
# In your project, create a file at .streamlit/secrets.toml
# and add your keys like this:
# HUGGINGFACEHUB_API_TOKEN = "your_huggingface_token"
# COHERE_API_KEY = "your_cohere_api_key"

os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
os.environ["COHERE_API_KEY"] = st.secrets.get("COHERE_API_KEY", "")

# --- Database Functions ---
@st.cache_resource
def init_connection():
    """Initializes and returns a connection to the SQLite database."""
    return sqlite3.connect("quiz_results.db", check_same_thread=False)

def create_table_if_not_exists(conn):
    """Creates the quiz_performance table if it doesn't already exist."""
    cursor = conn.cursor()
    create_table_query = """
    CREATE TABLE IF NOT EXISTS quiz_performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        class_name TEXT NOT NULL,
        chapter_name TEXT NOT NULL,
        total_questions INTEGER NOT NULL,
        correct_answers INTEGER NOT NULL,
        wrong_answers INTEGER NOT NULL,
        efficiency REAL NOT NULL,
        quiz_date DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
    cursor.execute(create_table_query)
    conn.commit()

def insert_quiz_data(conn, data_list, class_name):
    """Inserts quiz performance data into the database."""
    cursor = conn.cursor()
    data_list = data_list.replace('```json\n', '').replace('\n```','')
    data_list = json.loads(data_list)
    for item in data_list:
        chapter_name = item.get("chapter_name", "Unknown Chapter")
        total_questions = item.get("total_questions", 0)
        correct_answers = item.get("correct_answers", 0)
        
        if total_questions > 0:
            wrong_answers = total_questions - correct_answers
            efficiency = (correct_answers / total_questions) * 100
            insert_query = """
            INSERT INTO quiz_performance (class_name, chapter_name, total_questions, correct_answers, wrong_answers, efficiency, quiz_date)
            VALUES (?, ?, ?, ?, ?, ?, date('now'));
            """
            cursor.execute(insert_query, (class_name, chapter_name, total_questions, correct_answers, wrong_answers, efficiency))
    conn.commit()

# def get_latest_efficiency(conn, chapter_name):
#     """Retrieves the latest efficiency score for a specific chapter."""
#     cursor = conn.cursor()
#     placeholders = ','.join('?' for _ in chapter_name)
#     query = f"""
#     SELECT chapter_name, efficiency FROM quiz_performance
#     WHERE chapter_name IN ({placeholders})
#       AND quiz_date = (
#           SELECT MAX(quiz_date)
#           FROM quiz_performance AS sub
#           WHERE sub.chapter_name = quiz_performance.chapter_name
#       )
#     """
#     cursor.execute(query, chapter_name)
#     rows = cursor.fetchall()
    
#     # Convert to dictionary for easy lookup
#     return {chapter: eff for chapter, eff in rows}

def get_latest_efficiency(conn, chapter_name):
    cursor = conn.cursor()

    # Ensure it's always a list
    if isinstance(chapter_name, str):
        chapter_name = [chapter_name]

    placeholders = ','.join('?' for _ in chapter_name)
    query = f"""
    SELECT chapter_name, efficiency FROM quiz_performance
    WHERE chapter_name IN ({placeholders})
      AND quiz_date = (
          SELECT MAX(quiz_date)
          FROM quiz_performance AS sub
          WHERE sub.chapter_name = quiz_performance.chapter_name
      )
    """
    cursor.execute(query, chapter_name)
    rows = cursor.fetchall()

    return {chapter: eff for chapter, eff in rows}

def get_performance_data(conn):
    """Fetches all performance data for visualization."""
    query = "SELECT class_name, chapter_name, efficiency, quiz_date FROM quiz_performance ORDER BY quiz_date DESC;"
    df = pd.read_sql_query(query, conn)
    return df

# --- Vector Store and Embeddings ---
@st.cache_resource
def get_embedding_model():
    """Loads and caches the sentence-transformer embedding model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vector_store(pdf_path, vector_store_path):
    """Creates and saves a FAISS vector store from a PDF."""
    if not os.path.exists(pdf_path):
        st.error(f"PDF file not found at {pdf_path}")
        return
        
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)
    embedding_model = get_embedding_model()
    vector_store = FAISS.from_documents(split_docs, embedding_model)
    vector_store.save_local(vector_store_path)
    st.success(f"Vector store created and saved at {vector_store_path}")

@st.cache_resource
def load_vector_store(vector_store_path):
    """Loads a FAISS vector store from a local path."""
    if not os.path.exists(vector_store_path):
        return None
    embedding_model = get_embedding_model()
    return FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)

# --- LLM and RAG Chain Functions ---
def format_docs(docs):
    """Formats a list of documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# Updated prompts for clarity and better LLM guidance
quiz_prompt_template = """
You are an expert quiz creator. Your task is to generate a 5-question multiple-choice quiz from the provided chapter content.

**Rules:**
1.  Generate exactly 5 MCQs.
2.  **Adaptive Difficulty:** Adjust the difficulty based on the student's last score.
    -   Low Score (<40%): 4 Easy, 1 Medium.
    -   Medium Score (40-70%): 2 Easy, 3 Medium.
    -   High Score (>70%): 1 Easy, 2 Medium, 2 Hard.
    -   No Score: All 5 questions should be Easy.
3.  Each question MUST be labeled with its difficulty (e.g., `Difficulty: Easy`) and the chapter name (e.g., `[Electrostats]`).
4.  Do NOT provide answers.

**Student's Last Score (Efficiency):** {latest_score}%
**Chapter:** {chapter}
**Content:**
{context}

**Generate the quiz now.**
"""

evaluation_prompt_template = """
You are an expert evaluator. Evaluate the student's answers for the given quiz.

**Quiz:**
{quiz}

**Student's Answers:**
{student_answers}

**Your Task:**
For each question, provide:
- The score (1 for correct, 0 for wrong).
- If the answer is wrong, provide the correct answer and a concise explanation (under 50 words).

**Format:**
1. Score: 1
2. Score: 0, Correct Answer: B, Explanation: ...

**Begin evaluation.**
"""

summary_prompt_template = """
You are a data extraction specialist. From the provided quiz and evaluation, identify the chapters and count the total and correctly answered questions for each.

*Quiz:*
{quiz}

*Evaluation:*
{evaluation}

*Output ONLY a valid JSON array with this exact structure:*
[
  {{
    "chapter_name": "Chapter Name",
    "total_questions": 5,
    "correct_answers": <number_of_correct_answers>
  }}
]
"""

qa_prompt_template = """
You are an expert teacher. Answer the student's question based ONLY on the provided context. If the context does not contain the answer, state that the information is not available in the textbook.

**Context:**
{context}

**Question:**
{question}

**Answer:**
"""

@st.cache_resource
def get_llm():
    """Initializes and caches the Cohere LLM."""
    if not os.environ["COHERE_API_KEY"]:
        st.error("COHERE_API_KEY not found. Please add it to your .streamlit/secrets.toml file.")
        return None
    return ChatCohere(model="command-r", temperature=0.1)


def generate_quiz(retriever, chapter_names, latest_score):
    llm = get_llm()
    if not llm: return ""

    cohere_llm = ChatCohere(
        model="command-xlarge-nightly",
        temperature=0.05,
        cohere_api_key=os.getenv('cohere_api_key')
    )
    
    # Ensure chapter_names is a list
    if isinstance(chapter_names, str):
        chapter_names = [chapter_names]
    
    # Retrieve content for all chapters
    context_texts = []
    for chapter in chapter_names:
        relevant_docs = retriever.get_relevant_documents(chapter)
        context_texts.append(format_docs(relevant_docs))

    combined_context = "\n\n".join(context_texts)
    
    prompt = PromptTemplate.from_template(quiz_prompt_template)
    rag_chain = ({"context": lambda x: combined_context, 
     "chapter": lambda x: chapter_names, 
     "latest_score": lambda x: x['latest_score']} |
        prompt | cohere_llm | StrOutputParser())
    
    return rag_chain.invoke({
        "context": combined_context,
        "chapter":  chapter_names,
        "latest_score": latest_score
    })

def evaluate_quiz(quiz_text, student_answers):
    cohere_llm = ChatCohere(
        model="command-xlarge-nightly",
        temperature=0.05,
        cohere_api_key=os.getenv('cohere_api_key')
    )

    eval_prompt = PromptTemplate.from_template(evaluation_prompt_template)

    eval_chain = (
        {"student_answers": RunnablePassthrough(), "quiz": lambda x: quiz_text}
        | eval_prompt
        | cohere_llm
        | StrOutputParser()
    )

    return eval_chain.invoke(student_answers)

def get_chapter_summary(quiz_text, evaluation_text):
    cohere_llm = ChatCohere(
        model="command-xlarge-nightly",
        temperature=0,
        cohere_api_key=os.getenv('cohere_api_key')
    )

    # Prepare prompt template with correct variable names
    prompt = PromptTemplate.from_template(summary_prompt_template)

    # Create chain
    chain = (
        {"quiz": RunnablePassthrough(), "evaluation": RunnablePassthrough()}
        | prompt
        | cohere_llm
        | StrOutputParser()
    )

    # Invoke LLM
    summary_json = chain.invoke({"quiz": quiz_text, "evaluation": evaluation_text})
    return summary_json

def generate_qna_answer(chapter_names, retriever, model="command-xlarge-nightly", num_questions=10):
    cohere_llm = ChatCohere(
        model='command-xlarge-nightly',
        temperature=0.05,
        cohere_api_key=os.getenv("cohere_api_key")
    )

    if isinstance(chapter_names, str):
        chapter_names = [chapter_names]

    # Retrieve context
    context_texts = []
    for chapter in chapter_names:
        relevant_docs = retriever.get_relevant_documents(chapter)
        context_texts.append("\n".join([doc.page_content for doc in relevant_docs]))
    combined_context = "\n\n".join(context_texts)
    chapters_text = ", ".join(chapter_names)

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """
        You are a question generator for Science.
        Using ONLY the context provided, generate exactly {num_questions} questions 
        from the chapters: {chapters}.
        
      
        
        Format:
        Q1. Question text
        Ans. ans upto 150 words
        
        
        Q2. Question text
        Ans. ans upto 150 words
        
        
        Context:
        {context}
        """
    )

    rag_chain = (
        {"context": lambda _: combined_context, "chapters": lambda _: chapters_text}
        | prompt.partial(num_questions=5)
        | cohere_llm
        | StrOutputParser()
    )

    return rag_chain.invoke({"context": combined_context, "chapters": chapters_text})

def generate_any_answer(question):
    cohere_llm = ChatCohere(
        model="command-xlarge-nightly",
        temperature=0.05,
        cohere_api_key=os.getenv('cohere_api_key')
    )
    prompt = PromptTemplate.from_template(template="You are an expert teacher. Answer the given query of the given question.")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | cohere_llm
        | StrOutputParser()
    )
    
    return rag_chain.invoke(question)

def call_llm(messages):
    response = ollama.chat(
        model="llama3.2:3b",  # change to the model you have
        messages=messages
    )
    return response["message"]["content"]
# --- Main Application UI ---

st.title("📚 Sahayak: Your Learning Duo")
st.markdown("Welcome! Select your class, take adaptive quizzes, and get answers from your textbook.")

# --- Database Initialization ---
conn = init_connection()
create_table_if_not_exists(conn)

# --- Sidebar for Configuration ---
st.sidebar.header("Configuration")
class_name = st.sidebar.selectbox("Select Your Class", ["Class 12", "Class 11", "Class 10", "Class 9",'Class 8', "Class 7", 'CLass 6'])

vector_store_folder = "vectorstores"
os.makedirs(vector_store_folder, exist_ok=True)

class_to_vector = {
    "Class 12": "science class 12",
    "Class 11": "science class 11",
    "Class 10": "NCERT-Class-10-Science",
    "Class 9": "class 9 science",
    "Class 8": 'science_class8',
    "Class 7": "Copy of science class 7",
    "Class 6": 'science_class6'
}

pdf_filename = class_to_vector.get(class_name, "")
vector_store_path = os.path.join(vector_store_folder, class_to_vector.get(class_name, ""))

st.sidebar.markdown("---")
# st.sidebar.subheader("Textbook Management")
# st.sidebar.info(f"Place your textbook `{pdf_filename}` in the `{pdf_folder}` directory to enable features.")

# if st.sidebar.button("Create / Update Vector Store"):
#     if os.path.exists(pdf_path):
#         with st.spinner("Processing PDF and creating vector store... This may take a moment."):
#             create_vector_store(pdf_path, vector_store_path)
#     else:
#         st.sidebar.error(f"File not found. Please add `{pdf_filename}` to the `{pdf_folder}` folder.")

retriever = None
vector_store = load_vector_store(vector_store_path)
if vector_store:
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    st.sidebar.success(f"Textbook for **{class_name}** is loaded!")
else:
    st.sidebar.warning(f"Textbook for {class_name} not found. Please create the vector store.")

# --- Main Content Tabs ---
tab1, tab2, tab3 = st.tabs(["📝 Adaptive Quiz", "❓ Q&A Generator", "📊 My Performance"])

with tab1:
    st.header("📝 Take an Adaptive Quiz")
    if retriever:
        chapter_name = st.text_input("Enter the chapter name to generate a quiz:", "Electrostats", key="quiz_chapter")
        
        if st.button("Generate Quiz"):
            latest_score = get_latest_efficiency(conn, chapter_name)
            st.session_state.latest_score = latest_score
            with st.spinner(f"Generating a quiz for '{chapter_name}'"):
                quiz_text = generate_quiz(retriever, chapter_name, latest_score)
                st.session_state.quiz_text = quiz_text
                st.session_state.chapter_name = chapter_name
        
        if "quiz_text" in st.session_state:
            st.subheader("Your Quiz")
            st.markdown(st.session_state.quiz_text)
            with st.form("answers_form"):
                student_answers = st.text_area("Enter your answers here (e.g., 1. A, 2. B):", height=150)
                submitted = st.form_submit_button("Submit and Evaluate")
                
                if submitted:
                    with st.spinner("Evaluating your answers..."):
                        evaluation_text = evaluate_quiz(st.session_state.quiz_text, student_answers)
                        st.session_state.evaluation_text = evaluation_text
                        
                        st.subheader("Evaluation Result")
                        st.markdown(evaluation_text)
                        
                        summary_data = get_chapter_summary(st.session_state.quiz_text, evaluation_text)
                        if summary_data:
                            insert_quiz_data(conn, summary_data, class_name)
                            st.success("Your performance has been recorded!")
    else:
        st.warning("Please create a vector store in the sidebar to generate quizzes.")

with tab2:
    st.header("❓ Q&A Generator")
    if retriever:
        chapter = st.text_input("Enter Chapter", key="qa_question")
        if st.button("Get Q&A", key="get_answer_btn"):
            if chapter:
                with st.spinner("Searching for the answer..."):
                    answer = generate_qna_answer(chapter, retriever)
                    st.markdown(answer)
            else:
                st.warning("Please enter a question.")
    else:
        st.warning("Please create a vector store in the sidebar to ask questions.")

with tab3:
    st.header("📊 My Performance")
    st.write("Review your quiz performance over time.")
    
    performance_df = get_performance_data(conn)
    
    if not performance_df.empty:
        st.dataframe(performance_df, use_container_width=True)
        
        st.subheader("Average Efficiency by Chapter")
        chart_df = performance_df.groupby("chapter_name")["efficiency"].mean().sort_values(ascending=False)
        st.bar_chart(chart_df)
    else:
        st.info("No performance data yet. Take a quiz to track your progress!")

    
