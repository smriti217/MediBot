from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os
import chromadb
from langchain.vectorstores import Chroma

load_dotenv()

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")

#Creating Embeddings for Each of The Text Chunks & storing
#docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
vectorstore = Chroma.from_documents(text_chunks, embeddings)