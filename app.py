from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings,load_pdf,text_split
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from src.prompt import *
import os
import chromadb
app = Flask(__name__)

load_dotenv()


chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")


embeddings = download_hugging_face_embeddings()

extracted_data=load_pdf("C:/Users/smrit/WS/data")
text_chunks = text_split(extracted_data)

#Loading the index
vectorstore = Chroma.from_documents(text_chunks, embeddings)


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})


qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)