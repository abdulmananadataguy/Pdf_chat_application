import streamlit as st 
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter  # to spilit the text 
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain                   #conversational_retrieval
from htmlTemplates import css,user_template,bot_template
import os




# lets create a fuction which pull al pdf and then pull all content from their pages and put all of that in

def get_pdf_text(pdf_docs):
        text=""
        for pdf in pdf_docs:
            Pdf_reader= PdfReader(pdf)
            for page in Pdf_reader.pages:
                text+=page.extract_text()
        return text

# let get chunks of content 

def get_text_chunks(text):
     text_splitter = CharacterTextSplitter(separator="/n",chunk_size=1000,chunk_overlap=200,length_function=len)
     chunks = text_splitter.split_text(text)
     return chunks

# lets build a vector store 

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectostore= FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectostore
     
# Let's build a conversation chain 

def get_conversation_chain(vectorstore):
         # it has 3 componnets vector store - llm - memory 
        llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"],model="gpt-4o-mini")
        memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
        conversation_chain=ConversationalRetrievalChain.from_llm(llm=llm,retriever=vectorstore.as_retriever(),memory=memory)
        return conversation_chain
     
# now build an input function when user gives an input : 
def get_user_input(user_question):
    response = st.session_state.conversation({"question":user_question})
    st.session_state.chat_history=response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:  # its a user and we use a user template
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        

     




def main():
    load_dotenv()
    # Lets build the userinterface 

    st.set_page_config(page_title="Chat with your Pdf :ooks:",page_icon="books:",layout="centered")
    # enter css here 

    st.write(css,unsafe_allow_html=True)

    if "conversation" not in st.session_state:  #converation is actullay conversatioal retrivel chain that combine llm+vectorstore+memeory 
         st.session_state.conversation=None
    if "chat_history" not in st.session_state:   # chat_hsitory actually memory which reocred all the chat istory 
         st.session_state.chat_history=None

    st.header("Chat with your Pdf :books:")

    user_question = st.text_input("Enter your question releated to the uploaded files here :Books:")

    if user_question:
         get_user_input(user_question)

    # now build a side bar: 
    with st.sidebar:  #with function is used when you want to enter something into the side bar
        st.subheader("Your Documents")
        
        pdf_docs =st.file_uploader("Upload your pdf files here :books:",accept_multiple_files=True) # now multiple files can be uploaded
        if st.button("Process"): 
            # I want to create a spinner wheel which kept spinning and behind the scence setp 1,2,3 will be get done and then it stops 
            # since i want that step 1,2,3 occcur when spiine is moving then i have to use "with" 
             
            with st.spinner("Processing"):

                raw_text = get_pdf_text(pdf_docs)

                text_chunks = get_text_chunks(raw_text)

                vector_store= get_vector_store(text_chunks)

                st.session_state.conversation =get_conversation_chain(vector_store)   # conversation 

                

        
    


if __name__=='__main__':
    main()