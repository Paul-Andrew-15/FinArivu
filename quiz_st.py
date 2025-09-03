import os
import re
import random
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import AIMessage

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", api_key=GOOGLE_API_KEY)
db = Chroma(persist_directory="chroma_store", embedding_function=embedding_model)
retriever = db.as_retriever()

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """
        You are a friendly and expert financial literacy quiz assistant.  
        Your task is to generate **10 multiple-choice questions** on a given topic in finance or investing using the provided context: {context}  

        Requirements:

        1. Questions & Options
        - Generate exactly 10 questions. Do not generate more or fewer.
        - Each question must have 4 options labeled: a, b, c, d.
        - Randomize the order of options for each question while keeping the correct answer accurate.
        - Clearly indicate the correct answer using: "Correct answer: <letter>"
        - Provide a beginner-friendly explanation using: "Explanation: <text>"

        2. Language & Tone
        - Keep questions simple and beginner-friendly.
        - Explanations should be short, clear, and encouraging.
        - Maintain financial terminology accurately.
        - Output must be plain text, following the format below. Do not add extra text or headings.

        3. Multilingual
        - Translate all questions, options, and explanations into the user's preferred language, which can be any of the following:  
        English (en), Hindi (hi), Bengali (bn), Telugu (te), Marathi (mr), Tamil (ta), Urdu (ur), Gujarati (gu), Kannada (kn), Malayalam (ml), Chinese (zh), Spanish (es), Arabic (ar), French (fr), Russian (ru), Portuguese (pt), Japanese (ja), German (de), Javanese (jv), Korean (ko), Vietnamese (vi), Turkish (tr), Italian (it), Thai (th), Dutch (nl), Greek (el), Swedish (sv), Hungarian (hu), Hebrew (he)
        - Maintain the correct formatting for multiple-choice questions.
        - Avoid including language-specific instructions or notes in the output.

        4. Format Example
        1. What is a stock?
        a) A type of bond
        b) Ownership in a company
        c) A loan to a bank
        d) A government certificate
        Correct answer: b
        Explanation: A stock represents ownership in a company and may pay dividends.

        Always follow this format exactly and generate exactly 10 questions. Respond in the language requested by the user.

        """
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}")
])


document_chain = create_stuff_documents_chain(llm, prompt_template, document_variable_name="context")
retrieval_chain = create_retrieval_chain(retriever, document_chain)

def parse_questions(quiz_text, topic=None):
    questions = []
    q_splits = re.split(r'^\d+\.\s+', quiz_text, flags=re.MULTILINE)
    if len(q_splits) <= 1:
        q_splits = quiz_text.split('\n\n')
    for q_text in q_splits:
        lines = [line.strip() for line in q_text.strip().split('\n') if line.strip()]
        if not lines:
            continue
        question_line = re.sub(r'^\*+\s*', '', lines[0])
        options = {}
        for line in lines[1:]:
            match = re.match(r'(?:\*?\s*)([a-dA-D])[).]\s*(.*)', line)
            if match:
                key, val = match.groups()
                options[key.lower()] = val.strip()
        correct = None
        explanation = ""
        for line in lines:
            ans_match = re.match(r'Correct answer[:\s]*([a-dA-D])', line, re.IGNORECASE)
            if ans_match:
                correct = ans_match.group(1).lower()
            expl_match = re.match(r'Explanation[:\s]*(.*)', line, re.IGNORECASE)
            if expl_match:
                explanation = expl_match.group(1).strip()
        if options and correct:
            keys, vals = list(options.keys()), list(options.values())
            combined = list(zip(keys, vals))
            random.shuffle(combined)
            shuffled_keys, shuffled_vals = zip(*combined)
            new_options = {k: v for k, v in zip(['a', 'b', 'c', 'd'], shuffled_vals)}
            new_correct = ['a', 'b', 'c', 'd'][[i for i, k in enumerate(shuffled_keys) if k == correct][0]]
            key_concept = topic if topic else question_line[:50]
            questions.append({
                'question': question_line,
                'options': new_options,
                'correct': new_correct,
                'explanation': explanation,
                'key_concept': key_concept
            })
    return questions

def validate_topic_vector(db, topic_input, embedding_model):
    topic_input = topic_input.strip().lower()
    if topic_input in ["exit", "bye"]:
        return "exit"
    try:
        topic_embedding = embedding_model.embed_query(topic_input)
        results = db.similarity_search_by_vector(topic_embedding, k=3)
        if not results:
            return None
        return topic_input
    except Exception:
        return None

st.title("📊 FinArivu - Quiz Spot")

language_map = {
    "English": "en","Hindi": "hi","Bengali": "bn","Telugu": "te","Marathi": "mr","Tamil": "ta",
    "Urdu": "ur","Gujarati": "gu","Kannada": "kn","Malayalam": "ml","Chinese": "zh","Spanish": "es",
    "Arabic": "ar","French": "fr","Russian": "ru","Portuguese": "pt","Japanese": "ja","German": "de",
    "Javanese": "jv","Korean": "ko","Vietnamese": "vi","Turkish": "tr","Italian": "it","Thai": "th",
    "Dutch": "nl","Greek": "el","Swedish": "sv","Hungarian": "hu","Hebrew": "he"
}

for key, default in [("lang_code", "en"), ("quiz_questions", []), ("user_answers", {}),
                     ("topic", None), ("chat_history", []), ("submitted", False)]:
    if key not in st.session_state:
        st.session_state[key] = default

selected_lang = st.selectbox("Select your preferred language:", list(language_map.keys()))
st.session_state.lang_code = language_map[selected_lang]

if st.session_state.topic is None:
    topic_input = st.text_input("Enter quiz topic:")
    if st.button("Start Quiz") and topic_input.strip():
        topic = validate_topic_vector(db, topic_input, embedding_model)
        if topic == "exit":
            st.info("Quiz session ended.")
            st.stop()
        elif topic is None:
            st.warning("⚠️ Topic not found in knowledge base.")
            st.stop()
        else:
            response = retrieval_chain.invoke({
                "input": f"Generate 10 multiple-choice questions with 4 options each on: {topic} in {selected_lang}",
                "chat_history": st.session_state.chat_history
            })
            quiz_text = response.get("answer", "").strip()
            if quiz_text:
                st.session_state.quiz_questions = parse_questions(quiz_text, topic=topic)
                st.session_state.topic = topic
                st.session_state.chat_history.append(AIMessage(content=quiz_text))

if st.session_state.quiz_questions and not st.session_state.submitted:
    with st.form("quiz_form"):
        st.subheader(f"Topic: {st.session_state.topic.capitalize()}")
        for i, q in enumerate(st.session_state.quiz_questions):
            st.markdown(f"**Q{i+1}. {q['question']}**")
            
            if i not in st.session_state.user_answers:
                st.session_state.user_answers[i] = None

            st.session_state.user_answers[i] = st.radio(
                f"Choose your answer for Q{i+1}:",
                list(q['options'].keys()),
                index=None, 
                format_func=lambda x, opts=q['options']: f"{x}) {opts[x]}",
                key=f"q{i}"
            )
        
        submitted = st.form_submit_button("Submit All Answers")
        if submitted:
            unanswered = [i+1 for i, ans in st.session_state.user_answers.items() if ans is None]
            if unanswered:
                st.warning(f"⚠️ Please select an option for all questions. Unanswered: {', '.join(map(str, unanswered))}")
            else:
                st.session_state.submitted = True

if st.session_state.submitted:
    total = len(st.session_state.quiz_questions)
    correct_count = 0
    for i, q in enumerate(st.session_state.quiz_questions):
        user_ans = st.session_state.user_answers.get(i)
        correct = q['correct']
        if user_ans == correct:
            st.success(f"✅ Q{i+1}: Correct! {q['options'][correct]}")
            correct_count += 1
        else:
            st.error(f"❌ Q{i+1}: Wrong. Your answer: {user_ans}) {q['options'].get(user_ans, '')} | Correct: {correct}) {q['options'][correct]}")
        st.info(f"💡 Explanation: {q['explanation']}")

    st.subheader(f"🎉 Final Score: {correct_count}/{total}")

    if st.button("Take another quiz"):
        for key in ["topic","quiz_questions","user_answers","submitted"]:
            st.session_state[key] = None if key=="topic" else [] if key=="quiz_questions" else {} if key=="user_answers" else False
