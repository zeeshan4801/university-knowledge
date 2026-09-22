import os
import json
import streamlit as st

from groq import Groq

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS



# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="University Knowledge Assistant",
    page_icon="🎓",
    layout="wide"
)


# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


FAISS_DIR = os.path.join(
    BASE_DIR,
    "university_rag_faiss"
)


CONFIG_PATH = os.path.join(
    BASE_DIR,
    "config.json"
)


# ==========================================
# LOAD CONFIG
# ==========================================

@st.cache_data
def load_config():

    if os.path.exists(CONFIG_PATH):

        with open(
            CONFIG_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)


    return {

        "embedding_model":
        "BAAI/bge-base-en-v1.5"

    }



config = load_config()



# ==========================================
# CHECK FAISS FILES
# ==========================================


faiss_file = os.path.join(
    FAISS_DIR,
    "index.faiss"
)


pickle_file = os.path.join(
    FAISS_DIR,
    "index.pkl"
)



if not os.path.exists(faiss_file):

    st.error(
        """
        ❌ FAISS database missing.

        Required files:

        university_rag_faiss/
            index.faiss
            index.pkl

        Upload these files to GitHub repository.
        """
    )

    st.stop()



# ==========================================
# EMBEDDING MODEL
# ==========================================

@st.cache_resource
def load_embeddings():

    embeddings = HuggingFaceEmbeddings(

        model_name=config[
            "embedding_model"
        ],

        model_kwargs={

            "device":"cpu"

        },

        encode_kwargs={

            "normalize_embeddings":True

        }

    )

    return embeddings



embeddings = load_embeddings()



# ==========================================
# LOAD FAISS DATABASE
# ==========================================


@st.cache_resource
def load_vector_database():


    db = FAISS.load_local(

        FAISS_DIR,

        embeddings,

        allow_dangerous_deserialization=True

    )


    return db



vector_db = load_vector_database()



# ==========================================
# GROQ CLIENT
# ==========================================


if "GROQ_API_KEY" not in st.secrets:

    st.error(
        "Missing GROQ_API_KEY in Streamlit Secrets"
    )

    st.stop()



client = Groq(

    api_key=
    st.secrets["GROQ_API_KEY"]

)



# ==========================================
# TITLE
# ==========================================


st.title(
    "🎓 University Academic Knowledge Assistant"
)


st.write(
"""
Ask questions from official university documents.
The assistant retrieves information using RAG.
"""
)



# ==========================================
# CHAT
# ==========================================


question = st.chat_input(
    "Ask your academic question..."
)



if question:


    with st.chat_message(
        "user"
    ):

        st.write(question)



    # Retrieve documents

    docs = vector_db.similarity_search(

        question,

        k=5

    )



    context = ""

    sources = []



    for doc in docs:


        context += (

            "\n\n"
            +
            doc.page_content

        )


        sources.append(

            {

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

            }

        )



    # Prompt

    prompt=f"""

You are a university academic assistant.

Answer only from the provided context.

If the answer is not available,
say:
"I could not find this information in the university documents."


Context:

{context}


Question:

{question}

"""



    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[

            {

            "role":"system",

            "content":
            "You answer academic questions accurately."

            },


            {

            "role":"user",

            "content":prompt

            }

        ],

        temperature=0.1

    )



    answer = (
        response
        .choices[0]
        .message
        .content
    )



    with st.chat_message(
        "assistant"
    ):


        st.markdown(
            answer
        )


        st.markdown(
            "### 📚 Sources"
        )


        unique=set()


        for s in sources:

            citation=(

                f"📄 {s['file']} "
                f"(Page {s['page']})"

            )


            if citation not in unique:

                st.write(
                    citation
                )

                unique.add(citation)
