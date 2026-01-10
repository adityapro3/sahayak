# 📘 Sahayak

Sahayak is an a streamlit based application which generates personalized quizzes and question answers on the basis of students past performance with the help of ncert based curriculum books. The current version works on class 6-10 science subjects. 

---

## ⚙️ Technologies Used

- Python
- Langchain (for vector database creation & query retreival)
- Streamlit (for UI and app execution)
- SQLite (for storing quiz results)
- Vector Store (for semantic or experimental data storage)
- ollama (for calling the llm)

## 📂 Project Structure
```
sahayak/
│
├── app.py # Main Streamlit application
├── quiz_results.db # SQLite database for quiz results
├── vectorstores/ # Directory for vector / embedding storage
└── README.md # Project documentation
```
---

## ▶️ How to Run the Project

### Step 1: Clone the Repository
```
git clone https://github.com/adityapro3/sahayak.git
cd sahayak
```
### Step 2: Install Required Libraries
```
pip install streamlit langchain ollama sqlite3 pandas re
```
### Step 3: Run the Streamlit Application
```
streamlit run app.py
```

---

## 🧪 Features

- User-friendly interface using Streamlit
- Quiz interaction and result tracking
- Local database storage using SQLite
- Vector storage support for future AI extensions

## 🔮 Future Scope

- Improve UI design
- Add user login system
- Cloud database integration
