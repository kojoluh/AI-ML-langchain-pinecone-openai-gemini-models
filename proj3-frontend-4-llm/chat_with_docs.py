import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
  import os
  _, extension = os.path.splitext(file)

  if extension == '.pdf':
    from langchain_community.document_loaders import PyPDFLoader
    print(f'Loading {file}')
    loader = PyPDFLoader(file)
  elif extension == '.docx':
    from langchain_community.document_loaders import Docx2txtLoader
    print(f'Loading {file}')
    loader = Docx2txtLoader(file)
  elif extension == '.txt':
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(file)
  else:
    print('Document format is not supported!')
    return None

  data = loader.load()
  return data
  
# Chunking Data 
def chunk_data(data, chunk_size=256, chunk_overlap=20):
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  chunks = text_splitter.split_documents(data)
  return chunks

# Create Embeddings
def create_embeddings(chunks):
  embeddings = OpenAIEmbeddings()
  vector_store = Chroma.from_documents(chunks, embeddings)
  return vector_store

# Ask and Get Answer function
def ask_and_get_answer(vector_store, q, k=3):
  from langchain.chains import RetrievalQA
  from langchain_openai import ChatOpenAI

  llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

  retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

  chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
  
  answer = chain.invoke(q)
  return answer

# Calculating cost
def calculate_embedding_cost(texts):
  import tiktoken
  enc = tiktoken.encoding_for_model('text-embedding-3-small')
  total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
  # check prices here: https://openai.com/pricing
  # print(f'Total Tokens: {total_tokens}')
  # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
  return total_tokens, (total_tokens / 1000 * 0.0004)

# clear chat history area
def clear_history():
  if 'history' in st.session_state:
    del st.session_state['history']

def save_uploadedfile(uploadedfile):
     with open(os.path.join("files",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to files".format(uploadedfile.name))

# main app starts here
if __name__ == "__main__":
  import os
  from dotenv import load_dotenv, find_dotenv
  load_dotenv(find_dotenv(), override=True)

  st.image('./files/img.png')
  st.subheader('LLM Question-Answering Application!')

  with st.sidebar:
    api_key = st.text_input('OpenAI API Key: ', type='password')
    if api_key:
      os.environ['OPENAI_API_KEY'] = api_key

    uploaded_file = st.file_uploader('Upload a file: ', type=['pdf', 'docx', 'txt'])
    chunk_size = st.number_input('Chunk size: ', min_value=100, max_value=2048, value=512, on_change=clear_history)
    k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
    add_data = st.button('Add Data', on_click=clear_history)

    if uploaded_file and add_data:
      with st.spinner('Reading, chunking and embeddings file ...'):
        bytes_data = uploaded_file.read()
        file_name = os.path.join('./files/', uploaded_file.name)
        with open(file_name, 'wb') as f:
          f.write(bytes_data)

        data = load_document(file_name)
        chunks = chunk_data(data, chunk_size=chunk_size)
        st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

        tokens, embedding_cost = calculate_embedding_cost(chunks)
        st.write(f'Embedding cost: ${embedding_cost:.4f}')

        vector_store = create_embeddings(chunks)

        st.session_state.vs = vector_store
        st.success('File uploaded, chunked and embedded successfully.')

  if 'vs' in st.session_state:
    q = st.text_input('Ask a question about the content of your file: ')

    if q:
      # if 'vs' in st.session_state:
      vector_store = st.session_state.vs
      st.write(f'k: {k}')
      answer = ask_and_get_answer(vector_store, q, k)
      st.text_area('LLM Answer:', value=answer['result'])


      st.divider()
      if 'history' not in st.session_state:
        st.session_state.history = ''
      value = f'Q: {q}\nA: {answer}'
      st.session_state.history = f'{value} \n {"-" * 100}\n {st.session_state.history}'
      h = st.session_state.history
      st.text_area('Chat History:', value=h, key='history', height=400)
