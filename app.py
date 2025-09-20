import os
from dotenv import load_dotenv
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

# Load Google Gemini API key from .env
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize models
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=google_api_key, temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Guardrails prompt
template = """
You are an expert assistant. Follow these rules:
1. Answer the question using the retrieved documents in the context.
2. If the context is irrelevant or insufficient, you may use your own knowledge to answer.
3. Indicate clearly whether your answer uses:
   - PDF context only
   - Your own knowledge
   - Both
4. Always cite the document source(s) if any information comes from the PDFs.

Context (from PDFs):
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# Load PDF, split, and embed
def load_and_embed(pdf_file):
    loader = PyPDFLoader(pdf_file.name)
    documents = loader.load()

    if not documents:
        return None

    # Split documents into chunks
    chunks = splitter.split_documents(documents)

    # Create FAISS vector store (in-memory)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# PDF QA function
def pdf_qa(pdf_file, question):
    if not pdf_file:
        return "Please upload a PDF first."

    vectorstore = load_and_embed(pdf_file)
    if vectorstore is None:
        return "No valid text found in PDF."

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Set up RetrievalQA with guardrails prompt
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # Pass all retrieved docs at once
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )

    # Run QA
    try:
        result = qa.invoke({"query": question})
        answer = result["result"]

        # Collect sources
        sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
        if sources:
            answer += f"\n\nSources: {', '.join(set(sources))}"

        return answer
    except Exception as e:
        return f"Error querying Gemini: {e}"

# Gradio app
iface = gr.Interface(
    fn=pdf_qa,
    inputs=[
        gr.File(label="Upload PDF", file_types=[".pdf"]),
        gr.Textbox(label="Ask a Question", placeholder="Type your question here")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="📄 PDF Q&A with Guardrails",
    description="Upload a PDF and ask any question. Gemini generates answers using retrieved context and its own knowledge."
)

if __name__ == "__main__":
    iface.launch(share=False, show_error=True)
