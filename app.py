import os
import json
import streamlit as st

from groq import Groq

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS



# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="University Knowledge Assistant",
    page_icon="🎓",
    layout="wide"
)



# =====================================================
# LOAD CONFIG
# =====================================================

@st.cache_data
def load_config():

    with open(
        "config.json",
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
        "metadata.json",
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)



metadata = load_metadata()



# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(

        model_name=config["embedding_model"],

        model_kwargs={
            "device":"cpu"
        },

        encode_kwargs={

            "normalize_embeddings":True

        }
    )



embeddings = load_embeddings()



# =====================================================
# LOAD FAISS DATABASE
# =====================================================

@st.cache_resource
def load_faiss():

    return FAISS.load_local(

        "university_rag_faiss",

        embeddings,

        allow_dangerous_deserialization=True

    )



vector_db = load_faiss()



# =====================================================
# GROQ CLIENT
# =====================================================

groq_client = Groq(

    api_key=os.environ.get(
        "GROQ_API_KEY"
    )

)



# =====================================================
# HEADER
# =====================================================

st.title(
    "🎓 University Academic Knowledge Assistant"
)


st.write(
"""
Ask questions about:

- Admissions
- Academic rules
- Student handbook
- Examination rules
- Academic calendar
- Prospectus
- Department information

Answers are generated from official university documents.
"""
)



# =====================================================
# CHAT MEMORY
# =====================================================

if "messages" not in st.session_state:

    st.session_state.messages=[]



for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )



# =====================================================
# USER INPUT
# =====================================================

question = st.chat_input(
    "Ask your question..."
)



if question:


    st.session_state.messages.append({

        "role":"user",

        "content":question

    })



    with st.chat_message("user"):

        st.markdown(question)



    # ==============================================
    # VECTOR SEARCH
    # ==============================================


    retrieved_docs = vector_db.similarity_search(

        question,

        k=5

    )



    context = ""

    sources=[]



    for doc in retrieved_docs:


        context += (

            "\n\n"
            + 
            doc.page_content

        )



        sources.append({

            "document":
                doc.metadata.get(
                    "document_title",
                    "Unknown"
                ),

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



    # ==============================================
    # GROQ PROMPT
    # ==============================================


    prompt=f"""

You are a university academic assistant.

Rules:

1. Answer only from provided context.
2. Do not invent information.
3. If information is unavailable say:
"I could not find this information in the university documents."


Context:

{context}


Question:

{question}

"""



    response = groq_client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[

            {

            "role":"system",

            "content":
            "You provide accurate university academic information."

            },

            {

            "role":"user",

            "content":prompt

            }

        ],

        temperature=0.1

    )



    answer = response.choices[0].message.content



    # ==============================================
    # SOURCE FORMAT
    # ==============================================


    source_text="\n\n### 📚 Sources\n"


    unique=[]


    for src in sources:


        item=(

            f"📄 {src['file']} "
            f"(Page {src['page']})"

        )


        if item not in unique:

            unique.append(item)



    source_text += "\n".join(unique)



    final_answer = (

        answer

        +

        source_text

    )



    with st.chat_message("assistant"):

        st.markdown(final_answer)



    st.session_state.messages.append({

        "role":"assistant",

        "content":final_answer

    })
