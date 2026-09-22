import os
import json
import streamlit as st


from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from groq import Groq



# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(

    page_title="University Academic Assistant",

    page_icon="🎓",

    layout="wide"

)



# ==================================================
# LOAD CONFIG
# ==================================================

with open("config.json","r") as f:

    config=json.load(f)



EMBEDDING_MODEL=config["embedding_model"]



# ==================================================
# LOAD METADATA
# ==================================================

with open(
    "metadata.json",
    "r",
    encoding="utf-8"
) as f:

    metadata=json.load(f)



# ==================================================
# LOAD EMBEDDING MODEL
# ==================================================

@st.cache_resource

def load_embedding():

    return HuggingFaceEmbeddings(

        model_name=EMBEDDING_MODEL,

        model_kwargs={

            "device":"cpu"

        },

        encode_kwargs={

            "normalize_embeddings":True

        }

    )



embedding_model=load_embedding()



# ==================================================
# LOAD FAISS
# ==================================================

@st.cache_resource

def load_vector_db():

    return FAISS.load_local(

        "university_rag_faiss",

        embedding_model,

        allow_dangerous_deserialization=True

    )



vector_db=load_vector_db()



# ==================================================
# GROQ CLIENT
# ==================================================

client = Groq(

    api_key=os.environ.get(
        "GROQ_API_KEY"
    )

)



# ==================================================
# HEADER
# ==================================================

st.title(
    "🎓 University Academic Knowledge Assistant"
)


st.write(
"""
Ask questions related to:
- Admission
- Academic rules
- Student handbook
- Calendar
- Prospectus
- Department information
"""
)



# ==================================================
# CHAT MEMORY
# ==================================================

if "messages" not in st.session_state:

    st.session_state.messages=[]



for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )



# ==================================================
# USER QUERY
# ==================================================

question = st.chat_input(
    "Ask your academic question..."
)



if question:


    st.session_state.messages.append({

        "role":"user",

        "content":question

    })



    with st.chat_message("user"):

        st.write(question)



    # ----------------------------------------------
    # RETRIEVE DOCUMENTS
    # ----------------------------------------------


    docs = vector_db.similarity_search(

        question,

        k=5

    )



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
                    "source_file"
                ),

            "page":
                doc.metadata.get(
                    "page_number"
                ),

            "title":
                doc.metadata.get(
                    "document_title"
                )

        })



    # ----------------------------------------------
    # GROQ PROMPT
    # ----------------------------------------------


    prompt=f"""

You are a university academic assistant.

Answer only using the provided context.

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
            "You answer university questions accurately."

            },

            {

            "role":"user",

            "content":prompt

            }

        ],

        temperature=0.1

    )



    answer=response.choices[0].message.content



    # ----------------------------------------------
    # DISPLAY ANSWER
    # ----------------------------------------------


    final_answer = answer + "\n\n### 📚 Sources\n"


    unique_sources=[]


    for s in sources:

        source_text=(

            f"📄 {s['file']} "
            f"(Page {s['page']})"

        )


        if source_text not in unique_sources:

            unique_sources.append(
                source_text
            )



    final_answer += "\n".join(
        unique_sources
    )



    with st.chat_message("assistant"):

        st.write(final_answer)



    st.session_state.messages.append({

        "role":"assistant",

        "content":final_answer

    })
