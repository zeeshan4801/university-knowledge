import os
import json
import streamlit as st

from groq import Groq

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS



st.set_page_config(
    page_title="University Assistant",
    page_icon="🎓"
)



# Load config

with open("config.json") as f:

    config=json.load(f)



# Embeddings

@st.cache_resource
def embeddings():

    return HuggingFaceEmbeddings(

        model_name=config["embedding_model"],

        model_kwargs={
            "device":"cpu"
        },

        encode_kwargs={
            "normalize_embeddings":True
        }

    )


embed_model=embeddings()



# FAISS

@st.cache_resource
def load_db():

    return FAISS.load_local(

        "university_rag_faiss",

        embed_model,

        allow_dangerous_deserialization=True

    )


db=load_db()



# Groq

client=Groq(

    api_key=st.secrets["GROQ_API_KEY"]

)



st.title(
"🎓 University Academic Assistant"
)



question=st.chat_input(
"Ask your question"
)



if question:


    docs=db.similarity_search(

        question,

        k=5

    )


    context=""


    sources=[]


    for doc in docs:


        context+=doc.page_content+"\n"


        sources.append(

            f"{doc.metadata['source_file']} "
            f"(Page {doc.metadata['page_number']})"

        )



    prompt=f"""

Answer only from this context.

Context:

{context}


Question:

{question}

"""



    response=client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[

        {
        "role":"user",
        "content":prompt
        }

        ],

        temperature=0.1

    )


    answer=response.choices[0].message.content



    st.write(answer)


    st.subheader("Sources")

    for s in set(sources):

        st.write("📄",s)
