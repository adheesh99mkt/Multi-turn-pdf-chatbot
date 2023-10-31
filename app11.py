import streamlit as st
import pickle
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from htmlTemplate import css,bot_template,user_template
import os

# storing history in session states
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state["messages"]:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with  st.chat_message("assistant"):
            st.markdown(message["content"])

# Sidebar contents
with st.sidebar:
    st.header("upload here!")
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    

def main(pdf):
    load_dotenv()
    #st.set_page_config(page_title="chat with multiple PDFs",page_icon=":books:")
    st.header("Multi turn chatBot")
    st.write(css,unsafe_allow_html=True)
    if pdf is not None:
        # pdf reader
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # embeddings
        store_name = pdf.name[:-4]
        

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        # Accept user questions/query
        query = st.chat_input(placeholder="Ask questions about your PDF file:")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        if query:
            chat_history = []
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})

            custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. At the end of standalone question add this 'Answer the question in English language.' If you do not know the answer reply with 'I am sorry'.
                        Chat History:
                        {chat_history}
                        Follow Up Input: {question}
                        Standalone question:
                        Remember to greet the user with hi welcome to pdf chatbot how can i help you? if user asks hi or hello """

            CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

            llm = ChatOpenAI()
           
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm,
                VectorStore.as_retriever(),
                condense_question_prompt=CUSTOM_QUESTION_PROMPT,
                memory=memory
            )
            response = conversation_chain({"question": query, "chat_history": chat_history})
            # st.write(response["answer"])
            with st.chat_message("assistant"):
                st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            chat_history.append((query, response))

            


if __name__ == '__main__':
    main(pdf)