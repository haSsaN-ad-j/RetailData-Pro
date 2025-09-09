from sqlalchemy import create_engine
import pandas as pd
from google import genai
import re
import gradio as gr
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import requests
from bs4 import BeautifulSoup
import torch
import nltk

nltk.download('punkt', download_dir=r"C:\Users\Admin\AppData\Roaming\nltk_data")

# ---------------------------
# SETUP
# ---------------------------
nltk.download("punkt")

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

BERT_MODEL_NAME = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = AutoModelForQuestionAnswering.from_pretrained(BERT_MODEL_NAME)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = bert_model.to(device)
model = model.to(device)
embedding_model = embedding_model.to(device)

engine = create_engine("mysql+mysqlconnector://root:HASSAN123321hassan@127.0.0.1:3307/supermarket")
client = genai.Client(api_key="AIzaSyD4IcFq0ctSfJcocvc_5E5IDN6OEq9aeKg")  # put your key here

# ---------------------------
# SQL HELPER FUNCTIONS
# ---------------------------
def get_sql_query(question):
    prompt = f"""You are an expert SQL generator.
Database schema:
- Customers(customer_id, first_name, last_name, email, phone, address, created_at)
- Categories(category_id, category_name, description)
- Products(product_id, product_name, category_id, price, stock_quantity, description)
- Suppliers(supplier_id, supplier_name, contact_name, phone, email, address)
- Orders(order_id, customer_id, order_date, total_amount, status)
- OrderDetails(order_detail_id, order_id, product_id, quantity, price)
- Inventory(inventory_id, product_id, supplier_id, supply_date, quantity, cost_price)

Write a valid MySQL query for the question: {question}
"""
    response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
    sql_query = response.candidates[0].content.parts[0].text
    match = re.search(r"```sql(.*?)```", sql_query, re.DOTALL)
    if match:
        sql_query = match.group(1).strip()
    return question, sql_query

def sql_answer(sql_query):
    with engine.connect() as conn:
        df = pd.read_sql(sql_query, conn)
    return df

def generate_answer(question, answer):
    prompt = f"Generate a human response for this question: {question} with this answer: {answer}"
    response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
    return response.candidates[0].content.parts[0].text.strip()

def ask_question(question):
    q, sql = get_sql_query(question)
    try:
        df = sql_answer(sql)
    except Exception as e:
        return f"Error running SQL: {e}"
    return generate_answer(question, df.iloc[0,0] if not df.empty else "No results")

# ---------------------------
# PDF FUNCTIONS
# ---------------------------
def pdf_loader(pdf_file):
    reader = PdfReader(pdf_file)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            text_parts.append(page_text)
    return "\n".join(text_parts)

def clean_text_final(text: str) -> str:
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunking(text, chunk_size=700, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = start + chunk_size - overlap
    return chunks

def embed_text(chunks):
    return model.encode(chunks, show_progress_bar=True)

def pdf_retriever(query, chunks, embeddings, top_k=3):
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
    else:
        embeddings = embeddings.to(device)

    query_emb = model.encode(query, convert_to_tensor=True).to(device)
    cos_scores = util.cos_sim(query_emb, embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(top_k, len(chunks)))
    return [chunks[i] for i in top_results.indices.tolist()]

# ---------------------------
# WEBSITE FUNCTIONS
# ---------------------------
def process_url(url_link):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url_link, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 30]
    return " ".join(paragraphs) if paragraphs else soup.get_text(separator=" ", strip=True)

def web_retriever(chunks, question, k=3):
    embeddings = embedding_model.encode(chunks, convert_to_tensor=True).to(device)
    query_vec = embedding_model.encode([question], convert_to_tensor=True).to(device)
    cos_scores = util.cos_sim(query_vec, embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(k, len(chunks)))
    return [chunks[i] for i in top_results.indices.tolist()]

# ---------------------------
# DECISION + ANSWER
# ---------------------------
def answer_question(pdf_file, url, question):
    pdf_text = pdf_loader(pdf_file) if pdf_file else ""
    pdf_chunks = chunking(clean_text_final(pdf_text)) if pdf_text else []
    pdf_embeddings = embed_text(pdf_chunks) if pdf_chunks else []
    retriever_pdf = pdf_retriever(question, pdf_chunks, pdf_embeddings) if pdf_chunks else []

    website_text = process_url(url) if url else ""
    website_chunks = chunking(clean_text_final(website_text)) if website_text else []
    retriever_website = web_retriever(website_chunks, question) if website_chunks else []

    # Decide
    prompt = f"""
Choose the best source for the question: {question}
Options: sql, pdf, website, multiple
Instructions: 
- Answer ONLY with one word: sql, pdf, website, or multiple.
- Do NOT add any extra text.

"""
    response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
    decision = response.candidates[0].content.parts[0].text.strip().lower()

    if decision == "sql":
        return ask_question(question)
    elif decision == "pdf":
        return generate_answer(question, retriever_pdf)
    elif decision == "website":
        return generate_answer(question, retriever_website)
    elif decision == "multiple":
        return generate_answer(question, {"pdf": retriever_pdf, "website": retriever_website, "sql": ask_question(question)})
    if decision not in ["sql", "pdf", "website", "multiple"]:
      if pdf_chunks:  
        decision = "pdf"
      elif website_chunks:
        decision = "website"
      else:
        decision = "sql"

# ---------------------------
# GRADIO APP
# ---------------------------
def process_inputs(pdf_file, url, question):
    try:
        return answer_question(pdf_file, url, question)
    except Exception as e:
        return f"❌ Error: {str(e)}"

with gr.Blocks(title="AI Multi-Source Q&A") as demo:
    gr.Markdown("## 📚🌐🗄️ Ask Questions from PDF, Website, or Database")

    with gr.Row():
        pdf_input = gr.File(label="📄 Upload PDF", file_types=[".pdf"])
        url_input = gr.Textbox(label="🌍 Website URL")

    question_input = gr.Textbox(label="❓ Your Question")
    output_box = gr.Textbox(label="🤖 Answer", lines=10)

    ask_btn = gr.Button("🔎 Ask")
    ask_btn.click(fn=process_inputs, inputs=[pdf_input, url_input, question_input], outputs=output_box)

demo.launch(debug=True)










