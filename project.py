import streamlit as st
import os
import re
import time
from operator import itemgetter
from pathlib import Path
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from gtts import gTTS
import io
import base64
import hashlib 


TTS_CACHE_DIR = Path("tts_cache/") 
TTS_CACHE_DIR.mkdir(exist_ok=True) 
MAX_RETRIES = 3 
RETRY_DELAY = 5 



CHROMA_DB_DIR = "chroma_db/"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
GENERATION_MODEL = "gemini-2.5-flash"
WORD_DATA_SECRET_KEY = "WORD_DATA_CONTENT" 
RATE_LIMIT_SECONDS = 3 


try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    if os.getenv("GEMINI_API_KEY"):
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    else:
        st.error("ERROR: GEMINI API KEY not found. Please check Streamlit Secrets or .env file.")
        st.stop()
        
if not GEMINI_API_KEY:
    st.error("ERROR: GEMINI API KEY could not be loaded.")
    st.stop()

try:
    hf_token = st.secrets.get("HF_TOKEN")
    if hf_token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
except Exception:
    pass



def check_rate_limit(session_state):
    """
    Checks if the minimum time between queries (RATE_LIMIT_SECONDS) has passed.
    Returns (True, 0) if allowed, or (False, remaining_time) if restricted.
    """
    current_time = time.time()
    last_query_time = session_state.get("last_query_time", 0)

    if current_time - last_query_time < RATE_LIMIT_SECONDS:
        remaining_time = RATE_LIMIT_SECONDS - (current_time - last_query_time)
        return False, remaining_time
    
    session_state["last_query_time"] = current_time
    return True, 0

def check_prompt_relevance(prompt: str) -> bool:
    """
    Emulates the 'OffTopicPrompts' guardrail.
    Checks if the query contains generic keywords unrelated to translation/language learning.
    Returns True if the prompt is relevant, False if it is off-topic.
    """
    irrelevant_keywords = [
        "dünyanın en yüksek dağı", "nasıl kod yazılır", "hava durumu", 
        "kimsin", "kendini tanıt", "tarih", "matematik", "siyaset", 
        "tarif", "yemek tarifi", "haberler", "gündem", "şiir yaz"
    ]
    
    is_word_or_translation_query = any(q in prompt.lower() for q in [
        "translate", "çevir", "anlamı", "ne demek", "nedir", "kelime", "cümle"
    ])
    
    if len(prompt.split()) > 5 and not is_word_or_translation_query:
        if any(keyword in prompt.lower() for keyword in irrelevant_keywords):
            return False
            
    if len(prompt.split()) > 100:
         return False 

    return True



RAG_SYSTEM_PROMPT = """
You are an expert assistant designed for language learning. Your task is to provide the meaning, **ABSOLUTELY THE LEVEL INFORMATION**, and sample sentences for the word queried by the user, **OR** to provide a concise translation for a longer sentence query.

***TASK RULES***

1.**QUERY TYPE CHECK:**
  a. **WORD/MEANING QUERY:** If the user's input is a single word or a query asking for the meaning/level of a word (e.g., "what is 'apple'", "elma ne demek"), proceed with RAG and follow Rule 2.
  b. **SENTENCE/PHRASE QUERY (NEW):** If the user's input is a full sentence or a longer phrase (generally more than 5 words) that sounds like a translation request, provide only the **direct and concise translation** (English to Turkish or Turkish to English) without applying the RAG context or word format rules.

2.**CONTEXT PRIORITY (RAG for WORD QUERIES ONLY):**
  a. If you find a **DIRECT** and **RELIABLE** match for the keyword in the user's query within the 'Context' provided to you, generate your word meaning answer using the information and level from this Context.
  b. **LEVEL INFORMATION PRESENTATION (CRITICAL):** Always identify the **'| Level: [Level]'** information explicitly stated in the Context and **MUST** include it in your answer. If the Context does not contain level information (i.e., it's a general translation), **DO NOT INCLUDE** the level information.

3.**LANGUAGE CHECK & FORMAT (Mandatory for WORD QUERIES):** Always start the answer with the format appropriate for the language of the user's query.
  **If the query is English,** use the **"English Word Format."**
  **If the query is Turkish,** use the **"Turkish Word Format."**
  
4.**Sentence Rule (for WORD QUERIES):** For every word you select, construct **at least 3 different correct sample sentences independent of the context** that demonstrate different usage tones of the word. (Sentences must be in the target meaning language, i.e., English sentences for English meaning.)

---
Context:
{context}
---
Question: {input}

***
Sample Answer Formats (Only for Word/Meaning Queries):

# 1. English Word Format (Used if the query is English):
### [English Word] (Level: [Level Information, Exp: A1])
- **Turkish meaning:** [Meaning/Meanings]
- **Sample sentences:**
    1. [English Sentence 1]
    2. [English Sentence 2]
    3. [English Sentence 3]

# 2. Turkish Word Format (Used if the query is Turkish):
### [Turkish Word]
- **English meaning:** [Meaning/Meanings] (Level: [Level Information, Exp: A1])
- **Sample sentences:**
    1. [English Sentence 1]
    2. [English Sentence 2]
    3. [English Sentence 3]
***
"""


@st.cache_resource
def index_data(data_content: str, db_dir: str):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    if Path(db_dir).exists():
        try:
            vectorstore = Chroma(
                persist_directory=db_dir, 
                embedding_function=embeddings
            )
            
            if vectorstore._collection.count() > 0:
                return vectorstore
        except Exception as e:
            print(e) 
            pass
            
    
    with st.spinner("The system is being created, this may take some time, please wait."):
        
        splits = []
        if data_content:
            for i, line in enumerate(data_content.splitlines()):
                cleaned_line = line.strip()
                if cleaned_line:
                    splits.append(
                        Document(
                            page_content=cleaned_line,
                            metadata={"source": "Custom Dataset", "line": i + 1}
                        )
                    )
        
        if not splits:
            raise ValueError(f"The data content from Streamlit Secrets is empty or unreadable.")
            
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings,
            persist_directory=db_dir 
        )
        
    return vectorstore

def format_context_with_level(docs: list[Document]) -> str:
    """
    Processes documents (docs) coming from the Retriever.
    Captures level information (A1, B2, etc.), separates it from the content, and
    creates an explicitly labeled format for the model to see: [CONTENT] | Level: [LEVEL]
    """
    formatted_docs = []

    level_pattern = re.compile(r'\s+([A-C1-2]{1,2})$') 

    for i, doc in enumerate(docs):
        content = doc.page_content.strip()
        match = level_pattern.search(content)
        
        if match:
            level = match.group(1)
            cleaned_content = content[:match.start()].strip()
            formatted_line = f"[DATASET WORD {i+1}] {cleaned_content} | Level: {level}"
            formatted_docs.append(formatted_line)
        else:
            formatted_docs.append(f"[GENERAL CONTEXT {i+1}] {content}") 
            
    return "\n---\n".join(formatted_docs)

def is_rag_query(query_dict: dict) -> bool:    
    """
    Checks whether the query is a word translation/meaning/level query (RAG) 
    or a general question/sentence translation (NON-RAG/LLM-only).
    RAG is used for word lookups to include level information from the dataset.
    Sentence translation or general questions rely purely on the LLM.
    """
    query = query_dict["input"].lower()

    is_word_query_keywords = any(q in query for q in ["meaning of", "what is", "word", "level of", "give me", "translation of", "translate","ne demek","nedir","kelimesini","seviyesini","ver","anlamı","çevir"])
    is_short_query = len(query.split()) <= 5

    if is_word_query_keywords or (is_short_query and not is_word_query_keywords):
        return True
    
    return False


def extract_english_word(model_response: str, query: str) -> str:
    """
    It extracts the English word(s) to be vocalized from the model's response.
    It collects all English words from the titles (Format 1) or from the 'English meaning' lines (Format 2).
    
    It also checks if the response is a direct sentence translation 
    (i.e., not matching the word formats) and if the original query was Turkish, 
    it vocalizes the entire translated English sentence.
    """
    
    english_words_to_speak = set()
    
    for match_header in re.finditer(r"^###\s+([^#\n]+)\s*\(Level:\s*[A-C1-2]{1,2}\)", model_response, re.MULTILINE):
        header_content = match_header.group(1).strip()
        
        raw_english_words = re.sub(r'\s*\([^)]*\)', '', header_content).strip()
        
        words = re.findall(r'[a-zA-Z\s\-]+', raw_english_words)
        for w in words:
            clean_word = w.strip().lower()
            if clean_word and len(clean_word.split()) <= 5: 
                english_words_to_speak.add(clean_word)

    
    for match_meaning in re.finditer(r"^\s*-\s*\*\*English meaning:\*\*\s*([^(\n]+)", model_response, re.MULTILINE):
        meaning_text = match_meaning.group(1).strip()
        
        meaning_groups = re.split(r'[,;]+', meaning_text) 
        
        for group in meaning_groups:
            segment = group.strip()
            clean_segment = re.sub(r'[^a-zA-Z\s\-]', '', segment).strip() 
            
            if clean_segment:
                clean_word = clean_segment.lower()
                if len(clean_word.split()) <= 5: 
                    english_words_to_speak.add(clean_word)
    
    
    if not english_words_to_speak:
        if any(c in query for c in ['ı', 'ğ', 'ü', 'ş', 'ö', 'ç']):
            if re.match(r'^[a-zA-Z\s\.,\'"?!]+$', model_response.strip()):
                return model_response.strip().lower()

    if english_words_to_speak:
        return ", ".join(sorted(list(english_words_to_speak)))

    return ""


GENERAL_SYSTEM_PROMPT = "You are a helpful and knowledgeable assistant. Answer the user's general question concisely. If the input is a sentence translation request (e.g., 'elma almam gerekiyor'), provide ONLY the direct and concise translation (e.g., 'I need to buy an apple.') without any extra explanation or formatting."

@st.cache_resource
def create_rag_chain(_vectorstore: Chroma):
    with st.spinner(f"Establishing RAG Chain and Multiple Query ({GENERATION_MODEL})..."):
        
        llm = ChatGoogleGenerativeAI(
            model=GENERATION_MODEL, 
            temperature=0.7, 
            google_api_key=GEMINI_API_KEY
        )

        
        multiquery_retriever = MultiQueryRetriever.from_llm(
            retriever=_vectorstore.as_retriever(search_kwargs={"k": 3}),
            llm=llm
        )

        
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{input}")
        ])
        
        
        general_prompt = ChatPromptTemplate.from_messages([
            ("system", GENERAL_SYSTEM_PROMPT),
            ("human", "{input}")
        ])
        
        
        rag_chain_with_context_formatting = ( 
            RunnablePassthrough.assign(
                context=itemgetter("input") | multiquery_retriever | RunnableLambda(format_context_with_level) 
            )
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        
        final_chain = (
            RunnableBranch(
                (RunnableLambda(is_rag_query), rag_chain_with_context_formatting), 
                (itemgetter("input") | general_prompt | llm | StrOutputParser()),
            )
        )
        
    st.success("RAG Chain (with Dataset + General Translation Support) was established.") 
    return final_chain 

def main():
    st.set_page_config(page_title="Translation Bot", layout="wide")
    st.title("Dynamic RAG-Powered Translation Bot (with Sentence Translation and Voice-over)")
    
    data_content = None
    data_source_name = "Streamlit Secrets"
    
    if "data_storage" in st.secrets and WORD_DATA_SECRET_KEY in st.secrets["data_storage"]:
          data_content = st.secrets["data_storage"][WORD_DATA_SECRET_KEY]
          #st.caption(f"Data Source: **{data_source_name}** (`[data_storage]` table)")
    
    if not data_content:
        st.error(f"CRITICAL ERROR: The table **'[data_storage]'** or its key **'{WORD_DATA_SECRET_KEY}'** could not be found in Streamlit Secrets, or the content is empty. The application cannot be started. Please check your '.streamlit/secrets.toml' file.")
        return

    try:
        _vectorstore = index_data(data_content, CHROMA_DB_DIR)
        rag_chain = create_rag_chain(_vectorstore)
    except Exception as e:
        st.error(f"A critical error occurred during system installation (Indexing or RAG Chain creation): {e}")
        return

    
    info_text = f"""
    ## 📚 Welcome to your English-Turkish Language Teaching Assistant!
    
    This smart bot is designed for language learning and translation. It responds in two different modes depending on your query type:
    
    ### 1. 🔍 Vocabulary Queries (Level-Aware)
    * **Query:** The meaning, level, or usage of a word.
    * **Answer:** Includes **level information (A1, B2, etc.)**, its meaning, and 3 example sentences from our proprietary database.
    * **Example:** `What does execute mean?`
    
    ### 2. 💬 Sentence Translation (Quick Response)
    * **Query:** Translates a long sentence.
    * **Answer:** Provides a quick and direct translation, skipping RAG steps.
    * **Example:** `I don't want to be late for the meeting tomorrow.`
    
    
    ### ✨ Additional Features
    * **Read Aloud:** All English output is automatically converted to audio.
    
    ### 🛡️ Usage Notes
    * **Topic Restriction:** The bot only responds to questions related to **translation and language learning**. (General topics are blocked.)
    * **Speed Throttling:** To maintain API quota consumption, you must wait **{RATE_LIMIT_SECONDS} seconds** between consecutive queries.

    ---
    """
    
    st.markdown(info_text)


    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "last_query_time" not in st.session_state:
        st.session_state["last_query_time"] = 0


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            content = message["content"]
            st.markdown(content)
            
    if prompt := st.chat_input("Ask for a word or a sentence to be translated..."):
        
        is_allowed, remaining_time = check_rate_limit(st.session_state)
        if not is_allowed:
            st.warning(f"⚠️ **Too Fast!** Please wait **{remaining_time:.2f}** seconds before sending a new query to protect API usage.")
            return
        
        if not check_prompt_relevance(prompt):
            st.error("❌ **Prompt Blocked:** Your query is outside the scope of translation and language learning. Please ask questions related only to word meanings, sentence translations, or language assistance.")
            return
        
        timestamp = st.session_state.get("message_counter", 0) + 1
        st.session_state["message_counter"] = timestamp
        
        st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": timestamp})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            assistant_response_container = st.empty()
            with st.spinner("The answer is being sought and created..."):
                try:
                    input_data = {"input": prompt}
                    result = rag_chain.invoke(input_data)
                    
                    assistant_response_container.markdown(result)
                    
                    english_words_to_speak = extract_english_word(result, prompt)
                    
                    if english_words_to_speak:
                        cache_key = hashlib.sha256(english_words_to_speak.encode('utf-8')).hexdigest()
                        cache_file = TTS_CACHE_DIR / f"{cache_key}.mp3"

                        audio_bytes = None
                        
                        if cache_file.exists():
                            with open(cache_file, "rb") as f:
                                audio_bytes = f.read()
                        else:                            
                            tts_success = False

                            for attempt in range(MAX_RETRIES):
                                try:
                                    tts = gTTS(text=english_words_to_speak, lang='en', slow=False)
                                    mp3_fp = io.BytesIO()
                                    tts.write_to_fp(mp3_fp)
                                    mp3_fp.seek(0)
                                    
                                    audio_bytes = mp3_fp.read()
                                    with open(cache_file, "wb") as f:
                                        f.write(audio_bytes)
                                    
                                    tts_success = True
                                    break 
                                
                                except Exception as tts_e:
                                    error_message = str(tts_e)
                                    if "Too Many Requests" in error_message or "429" in error_message:
                                        if attempt < MAX_RETRIES - 1:
                                            st.warning(f"⚠️ **TTS API Restriction (429).** {attempt + 1}th attempt failed. Waiting for {RETRY_DELAY} seconds...")
                                            time.sleep(RETRY_DELAY)
                                        else:
                                            st.error(f"❌ Audio generation failed on {MAX_RETRIES} attempts. Detail: {tts_e}")
                                            break
                                    else:
                                        st.error(f"❌ Unexpected Audio Rendering Error: {tts_e}")
                                        break

                        if audio_bytes:
                            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                            
                           audio_html = f"""
                            <audio controls controlsList="nodownload" style="width: 150px; height: 30px;">
                                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                            </audio>
                            """
                            
                            st.markdown(audio_html, unsafe_allow_html=True)
                            
                            if len(english_words_to_speak.split()) > 5:
                                caption_text = "Vocalized sentence"
                            else:
                                caption_text = "Vocalized word(s)"
                                
                            st.caption(f"{caption_text}: **{english_words_to_speak[:100]}{'...' if len(english_words_to_speak) > 100 else ''}**")
                  
                    
                    st.session_state.messages.append({"role": "assistant", "content": result, "timestamp": timestamp})
                    
                except Exception as e:
                    error_msg = f"A critical error occurred while generating the response: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg, "timestamp": timestamp})

if __name__ == "__main__":
    main()
