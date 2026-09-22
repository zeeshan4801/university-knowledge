import os
import json
import streamlit as st

from groq import Groq

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS



# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="University Knowledge Assistant",
    page_icon="🎓",
    layout="wide"
)



# =====================================================
# PATH CONFIGURATION
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


FAISS_PATH = os.path.join(
    BASE_DIR,
    "university_rag_faiss"
)


CONFIG_FILE = os.path.join(
    BASE_DIR,
    "config.json"
)


METADATA_FILE = os.path.join(
    BASE_DIR,
    "metadata.json"
)



# =====================================================
# CHECK REQUIRED FILES
# =====================================================

required_files = [

    CONFIG_FILE,

    METADATA_FILE,

    os.path.join(
        FAISS_PATH,
        "index.faiss"
    ),

    os.path.join(
        FAISS_PATH,
        "index.pkl"
    )

]


for file in required_files:

    if not os.path.exists(file):

        st.error(
            f"Missing required file:\n\n{file}"
        )

        st.stop()



# =====================================================
# LOAD CONFIG
# =====================================================


@st.cache_data
def load_config():

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)



config = load_config()



# =====================================================
# LOAD METADATA
# =====================================================


@st.cache_data
def load_metadata():

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)



metadata = load_metadata()



# =====================================================
# LOAD EMBEDDINGS
# =====================================================


@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(

        model_name=config["embedding_model"],

        model_kwargs={

            "device":
            "cpu"

        },

        encode_kwargs={

            "normalize_embeddings":
            True

        }

    )



embeddings = load_embeddings()



# =====================================================
# LOAD FAISS DATABASE
# =====================================================


@st.cache_resource
def load_faiss_database():


    try:


        db = FAISS.load_local(

            FAISS_PATH,

            embeddings,

            allow_dangerous_deserialization=True

        )


        return db



    except Exception as e:


        st.error(
            "FAISS database could not be loaded."
        )


        st.write(
            "Check that index.faiss and index.pkl are uploaded correctly."
        )


        st.exception(e)

        st.stop()



vector_db = load_faiss_database()



# =====================================================
# GROQ CLIENT
# =====================================================


if "GROQ_API_KEY" not in st.secrets:

    st.error(
        "GROQ_API_KEY missing. Add it in Streamlit Secrets."
    )

    st.stop()



client = Groq(

    api_key=st.secrets["GROQ_API_KEY"]

)



# =====================================================
# HEADER
# =====================================================


st.title(
    "🎓 University Academic Knowledge Assistant"
)


st.markdown(
"""
Ask questions from official university documents.

Available sources:
- Student Handbook
- Academic Calendar
- Prospectus
- Examination Rules
- Department Information
"""
)



# =====================================================
# CHAT MEMORY
# =====================================================


if "messages" not in st.session_state:

    st.session_state.messages = []



for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.markdown(
            msg["content"]
        )



# =====================================================
# USER QUESTION
# =====================================================


question = st.chat_input(
    "Ask your academic question..."
)



if question:


    st.session_state.messages.append(

        {
            "role":"user",
            "content":question
        }

    )


    with st.chat_message("user"):

        st.markdown(question)



    # =================================================
    # RETRIEVAL
    # =================================================


    docs = vector_db.similarity_search(

        question,

        k=5

    )


    if len(docs)==0:


        context = (
            "No relevant information found."
        )


    else:

        context=""


        sources=[]


        for doc in docs:


            context += (

                "\n\n"
                +
                doc.page_content

            )


            sources.append({

                "file":
                doc.metadata.get(
                    "source_file",
                    "Unknown"
                ),


                "page":
                doc.metadata.get(
                    "page_number",
                    "Unknown"
                )

            })



    # =================================================
    # GROQ GENERATION
    # =================================================


    prompt=f"""

You are a university academic assistant.

Answer ONLY using the context below.

If information is unavailable say:

"I could not find this information in the university documents."


CONTEXT:

{context}


QUESTION:

{question}

"""



    completion = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[

            {

                "role":"system",

                "content":
                "You provide accurate academic answers."

            },


            {

                "role":"user",

                "content":prompt

            }

        ],

        temperature=0.1

    )



    answer = (
        completion
        .choices[0]
        .message
        .content
    )



    # =================================================
    # SOURCE CITATIONS
    # =================================================


    citation="\n\n### 📚 Sources\n"


    seen=set()


    for src in sources:


        item=(

            f"📄 {src['file']} "
            f"- Page {src['page']}"

        )


        if item not in seen:

            citation += item+"\n"

            seen.add(item)



    final_response = (

        answer

        +

        citation

    )



    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            final_response
        )



    st.session_state.messages.append(

        {

        "role":"assistant",

        "content":final_response

        }

    )
