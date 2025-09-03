import streamlit as st
import os
import re
import random
from dotenv import load_dotenv
from gtts import gTTS
import sounddevice as sd
import soundfile as sf
import io
import base64
import langid

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import HumanMessage, AIMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ----------------- Load Env -----------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ----------------- Streamlit Config -----------------
st.set_page_config(page_title="FinArivu", page_icon="💬")

# ----------------- Sidebar Navigation -----------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Chatbot", "Quiz"])

# ----------------- Shared Setup -----------------
SUPPORTED_LANGUAGES = [
    "en", "hi", "bn", "te", "mr", "ta", "ur", "gu", "kn", "ml",
    "zh", "es", "ar", "fr", "ru", "pt", "ja", "de", "jv", "ko",
    "vi", "tr", "it", "th", "nl", "el", "sv", "hu", "he"
]

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", api_key=GOOGLE_API_KEY)
db = Chroma(persist_directory="chroma_store", embedding_function=embedding_model)
retriever = db.as_retriever()

# ----------------- Helpers -----------------
def safe_detect(text: str) -> str:
    try:
        lang, _ = langid.classify(text)
        return lang
    except:
        return "en"

def speak(text, lang="en"):
    try:
        tts = gTTS(text=text, lang=lang)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        data, samplerate = sf.read(mp3_fp, dtype="float32")
        sd.play(data, samplerate)
        sd.wait()
    except Exception as e:
        print(f"TTS failed: {e}")

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
    except:
        return None

# ----------------- Chatbot Page -----------------
if page == "Chatbot":
    st.title("FinArivu - Smart Knowledge for Safer Investments")
    st.markdown(
    """
    <h2 style="color:#000; font-weight:bold;">
        💬 Chatbot
    </h2>
    """,
    unsafe_allow_html=True
)

    prompt_template_chat = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """
            You are a friendly and expert financial assistant chatbot focused on guiding individuals and families to improve their financial literacy and well-being.  
            You can securely store user details and financial goals for future reference.  

            Safety Note: Always provide accurate guidance, but remind users to consult a certified financial advisor for critical financial decisions.  

            Multilingual Support:  
            1. Users may ask questions in English, Hindi, Bengali, Telugu, Marathi, Tamil, Urdu, Gujarati, Kannada, Malayalam, Mandarin Chinese, Spanish, Arabic, French, Russian, Portuguese, Japanese, German, Javanese, Korean, Vietnamese, Turkish, Italian, Thai, Dutch, Greek, Swedish, Hungarian, or Hebrew.  
            2. Automatically detect the input language.  
            3. Translate the user’s query into English for internal understanding (embeddings are in English).  
            4. Generate your answer in the same language as the user’s input, naturally and culturally appropriate.  

            Your task:  
            1. Answer the user’s financial question using the provided context: {context}.  
            2. Focus on improving financial literacy, money management, savings, investments, loans, insurance, and planning, even if the user mentions personal or celebrity names.  
            3. Provide detailed, simple, and friendly explanations that are easy for beginners to understand.  
            4. Present answers in numbered lists for clarity when displayed as text.  
            Note: When the answer is read aloud via Text-to-Speech (TTS), do not use numbers, bullet points, or any formatting symbols; instead, speak naturally as flowing sentences.  
            5. If the question is vague (e.g., "I need money advice"), ask a human-like clarifying question in the same language.  
            6. If the question relates to timing, age, or financial milestones, provide approximate guidance in a clear and friendly manner.  
            7. Include 1–2 natural conversational prompts at the end to keep the discussion engaging.  
            8. If information is not available, respond: "Information not available." (translated into the user’s language).  
            9. If the question is completely unrelated, respond: "Please ask a valid question." (translated into the user’s language).  
            10. If the input language is not supported, politely inform the user.   
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    document_chain_chat = create_stuff_documents_chain(llm, prompt_template_chat, document_variable_name="context")
    retrieval_chain_chat = create_retrieval_chain(retriever, document_chain_chat)

    # Separate chatbot state
    if "chat_history_chatbot" not in st.session_state:
        st.session_state.chat_history_chatbot = []

    use_tts = st.checkbox("🔊 Read answer aloud (TTS)", value=False)

    # Display chatbot messages
    for msg in st.session_state.chat_history_chatbot:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    # User input
    user_input = st.chat_input("Type your question…")
    if user_input:
        lang_code = safe_detect(user_input)
        if lang_code not in SUPPORTED_LANGUAGES:
            lang_code = "en"

        st.session_state.chat_history_chatbot.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                inputs = {"input": user_input, "chat_history": st.session_state.chat_history_chatbot}
                response = retrieval_chain_chat.invoke(inputs)
                answer = response.get("answer", "").strip()
                st.markdown(answer)
                st.session_state.chat_history_chatbot.append(AIMessage(content=answer))
                if use_tts:
                    speak(answer, lang_code)

# ----------------- Quiz Page -----------------
elif page == "Quiz":
    st.title("FinArivu - Smart Knowledge for Safer Investments")
    st.markdown(
    """
    <h2 style="color:#000; font-weight:bold;">
        📊 Quiz Spot
    </h2>
    """,
    unsafe_allow_html=True
)

    prompt_template_quiz = ChatPromptTemplate.from_messages([
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
    document_chain_quiz = create_stuff_documents_chain(llm, prompt_template_quiz, document_variable_name="context")
    retrieval_chain_quiz = create_retrieval_chain(retriever, document_chain_quiz)

    language_map = {
        "English":"en","Hindi":"hi","Bengali":"bn","Telugu":"te","Marathi":"mr","Tamil":"ta",
        "Urdu":"ur","Gujarati":"gu","Kannada":"kn","Malayalam":"ml","Chinese":"zh","Spanish":"es",
        "Arabic":"ar","French":"fr","Russian":"ru","Portuguese":"pt","Japanese":"ja","German":"de",
        "Javanese":"jv","Korean":"ko","Vietnamese":"vi","Turkish":"tr","Italian":"it","Thai":"th",
        "Dutch":"nl","Greek":"el","Swedish":"sv","Hungarian":"hu","Hebrew":"he"
    }

    # Separate quiz state
    for key, default in [
        ("lang_code","en"),("quiz_questions",[]),("user_answers",{}),
        ("topic",None),("chat_history_quiz",[]),("submitted",False)
    ]:
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
                st.warning("⚠️ Topic not found.")
                st.stop()
            else:
                response = retrieval_chain_quiz.invoke({
                    "input": f"Generate 10 multiple-choice questions on: {topic} in {selected_lang}",
                    "chat_history": st.session_state.chat_history_quiz
                })
                quiz_text = response.get("answer", "").strip()
                if quiz_text:
                    st.session_state.quiz_questions = parse_questions(quiz_text, topic=topic)
                    st.session_state.topic = topic
                    st.session_state.chat_history_quiz.append(AIMessage(content=quiz_text))

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
                    st.warning(f"⚠️ Please answer all questions. Unanswered: {', '.join(map(str, unanswered))}")
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
