import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import HumanMessage, AIMessage
import langid
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from gtts.lang import tts_langs
from gtts import gTTS
import sounddevice as sd
import soundfile as sf
import io


load_dotenv()

st.set_page_config(page_title="FinArivu", page_icon="💬")

SUPPORTED_LANGUAGES = [
    "en", "hi", "bn", "te", "mr", "ta", "ur", "gu", "kn", "ml",
    "zh", "es", "ar", "fr", "ru", "pt", "ja", "de", "jv", "ko",
    "vi", "tr", "it", "th", "nl", "el", "sv", "hu", "he"
]

def safe_detect(text: str) -> str:
    try:
        lang, _ = langid.classify(text)
        return lang
    except Exception:
        return "unknown"

def translate_to_english(text: str, lang_code: str, llm) -> str:
    if lang_code == "en":
        return text.strip()
    prompt = f"""
Translate the following text from {lang_code} into plain, natural English.
Only return the translation, nothing else.

Text: "{text}"
""".strip()
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception:
        return text

def translate_from_english(text: str, lang_code: str, llm) -> str:
    if lang_code == "en":
        return text.strip()
    prompt = f"""
Translate the following English text into {lang_code}, naturally and fluently.
Only return the translated text, nothing else.

Text: "{text}"
""".strip()
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception:
        return text

GTTS_SUPPORTED_LANGS = tts_langs()

def speak(text, lang="en"):
    """Convert text to speech using gTTS and play directly without saving."""
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


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", google_api_key=os.getenv("GOOGLE_API_KEY"))

db = Chroma(
    persist_directory="chroma_store",
    embedding_function=embedding_model
)
retriever = db.as_retriever()

prompt_template = ChatPromptTemplate.from_messages([
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

document_chain = create_stuff_documents_chain(llm, prompt_template, document_variable_name="context")
retrieval_chain = create_retrieval_chain(retriever, document_chain)

st.title("FinArivu - Smart Knowledge for Safer Investments")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

use_tts = st.checkbox("🔊 Read answer aloud (TTS)", value=False)

with st.expander("ℹ️ How this works", expanded=False):
    st.markdown(
        """
- Ask questions about financial literacy in any supported language.  
- The chatbot will respond naturally in the same language.  
- Optionally, you can have the assistant read the answer aloud using Google TTS.
"""
    )

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

user_input = st.chat_input("Type your question…")

if user_input:
    lang_code = safe_detect(user_input)
    if lang_code not in SUPPORTED_LANGUAGES:
        lang_code = "en"

    english_query = translate_to_english(user_input, lang_code, llm) if lang_code != "en" else user_input

    inputs = {
        "input": english_query,
        "chat_history": st.session_state.chat_history
    }

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = retrieval_chain.invoke(inputs)
            llm_answer_en = response.get("answer", "").strip()
            final_answer = translate_from_english(llm_answer_en, lang_code, llm) if lang_code != "en" else llm_answer_en
            st.markdown(final_answer)
            st.session_state.chat_history.append(AIMessage(content=final_answer))

            if use_tts:
                tts_lang = lang_code if lang_code in SUPPORTED_LANGUAGES else "en"
                speak(final_answer, lang=tts_lang)


st.markdown("---")
st.caption("⚠️ Educational purposes only. For critical financial decisions, consult a certified financial advisor.")


