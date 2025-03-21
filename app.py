import boto3
# Using Titan Embedding model for embeddings
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
# Data ingestion
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
# Vector store
from langchain.vectorstores import FAISS
# LLM models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import tempfile
import os
import streamlit as st

# Initialize session state variables if they don't exist
if 'vectors_created' not in st.session_state:
    st.session_state.vectors_created = False
if 'response' not in st.session_state:
    st.session_state.response = ""

# Bedrock clients
bedrock = boto3.client(service_name='bedrock-runtime')
bedrock_embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v2:0', client=bedrock)

# Define prompt template for QA
prompt_template="""
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least 250 words to summarize 
with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# Process existing PDFs in the 'data' directory
def process_existing_pdfs():
    try:
        # Check if data directory exists
        if not os.path.exists('data'):
            st.error("The 'data' directory does not exist in the working directory.")
            return False
            
        # Check if there are PDF files in the directory
        pdf_files = [f for f in os.listdir('data') if f.lower().endswith('.pdf')]
        if not pdf_files:
            st.error("No PDF files found in the 'data' directory.")
            return False
            
        loader = PyPDFDirectoryLoader('data')
        docs = loader.load()
        
        # Split into smaller chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        documents = splitter.split_documents(docs)
        
        # Create vector embeddings
        vectorstore_faiss = FAISS.from_documents(
            documents,
            bedrock_embeddings
        )
        vectorstore_faiss.save_local('faiss_index')
        return True, len(pdf_files)
    
    except Exception as e:
        st.error(f"Error processing existing PDFs: {str(e)}")
        return False, 0

# Data Ingestion from uploaded PDFs
def process_uploaded_pdfs(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    try:
        all_docs = []
        for uploaded_file in uploaded_files:
            # Save the uploaded file to the temporary directory
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load the PDF
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            all_docs.extend(docs)
        
        # Split into smaller chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        documents = splitter.split_documents(all_docs)
        
        # Create vector embeddings
        vectorstore_faiss = FAISS.from_documents(
            documents,
            bedrock_embeddings
        )
        vectorstore_faiss.save_local('faiss_index')
        return True
    
    except Exception as e:
        st.error(f"Error processing PDFs: {str(e)}")
        return False
    finally:
        # Clean up temporary files
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)

# Llama 3 8B Instruct
def get_llama_llm():
    # Create the Llama LLM
    llm = Bedrock(model_id='meta.llama3-8b-instruct-v1:0', client=bedrock,
                model_kwargs={
                    'max_gen_len': 512                    
                })
    return llm

# Mistral 7B Instruct
def get_mistral_llm():
    bedrock_client = boto3.client(service_name='bedrock-runtime')
    
    llm = Bedrock(
        model_id='mistral.mistral-7b-instruct-v0:2',
        client=bedrock_client,
        model_kwargs={
            "max_tokens": 500
        }
    )
    return llm

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}  # top 3 results
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa.invoke({'query': query})
    return answer['result']

def clear_response():
    st.session_state.response = ""

# Streamlit Interface
def main():
    st.set_page_config(page_title="Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock 💁")
    
    # Sidebar for PDF upload and vector creation
    with st.sidebar:
        st.title("Upload or Use Demo PDFs")
        
        # Option to use existing PDFs in data directory
        st.subheader("Demo Option")
        if st.button("Process Existing PDFs in 'data' folder"):
            with st.spinner("Processing existing PDFs..."):
                success, num_files = process_existing_pdfs()
                if success:
                    st.session_state.vectors_created = True
                    st.success(f"Successfully processed {num_files} PDF(s) from 'data' directory")
                    st.info("You can now ask questions in the main panel")
        
        st.divider()
        
        # Option to upload new PDFs
        st.subheader("Upload Your PDFs")
        uploaded_files = st.file_uploader("Choose PDFs", accept_multiple_files=True, type=['pdf'])
        
        if uploaded_files:
            if st.button("Process Uploaded PDFs"):
                with st.spinner("Processing uploaded PDFs and creating vectors..."):
                    success = process_uploaded_pdfs(uploaded_files)
                    if success:
                        st.session_state.vectors_created = True
                        st.success(f"Successfully processed {len(uploaded_files)} uploaded PDF(s)")
                        st.info("You can now ask questions in the main panel")
                    else:
                        st.error("Failed to process uploaded PDFs")
    
    # Main content area
    if not st.session_state.vectors_created:
        st.info("👈 Please upload PDFs or use demo PDFs from the sidebar before asking questions")
    else:
        user_question = st.text_input("Ask a Question from the PDF Files")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Mistral Output"):
                if not user_question:
                    st.warning("Please enter a question")
                else:
                    with st.spinner("Processing with Mistral..."):
                        try:
                            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                            llm = get_mistral_llm()
                            st.session_state.response = get_response_llm(llm, faiss_index, user_question)
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")
        
        with col2:
            if st.button("Llama3 Output"):
                if not user_question:
                    st.warning("Please enter a question")
                else:
                    with st.spinner("Processing with Llama3..."):
                        try:
                            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                            llm = get_llama_llm()
                            st.session_state.response = get_response_llm(llm, faiss_index, user_question)
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")
        
        with col3:
            if st.button("Clear Response"):
                clear_response()
        
        # Display the response
        if st.session_state.response:
            st.markdown("### Response:")
            st.write(st.session_state.response)

if __name__ == "__main__":
    main()