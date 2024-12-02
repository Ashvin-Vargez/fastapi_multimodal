import streamlit as st
import base64
import os
import torch #additional to fix error 
import uuid
import io
from PIL import Image
import re
from typing import List, Dict
import tempfile

from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# Set page configuration
st.set_page_config(page_title="Multimodal RAG System", layout="wide")

# Initialize session state
if "processed_files" not in st.session_state:
    st.session_state.processed_files = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Utility functions
def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def image_summarize(img_base64: str, prompt: str) -> str:
    """Make image summary using GPT-4o"""
    chat = ChatOpenAI(model="gpt-4o", max_tokens=5000)
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content

def extract_pdf_elements(pdf_path: str, temp_dir: str):
    """Extract elements from PDF"""
    return partition_pdf(
        filename=pdf_path,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=temp_dir,
    )

def categorize_elements(raw_pdf_elements):
    """Categorize extracted elements from PDF"""
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
            print (tables) #for debugging
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    return texts, tables

def generate_text_summaries(texts: List[str], tables: List[str], summarize_texts: bool = False):
    """Generate summaries for texts and tables"""
    prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
    These summaries will be embedded and used to retrieve the raw text or table elements. \
    Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    
    model = ChatOpenAI(temperature=0, model="gpt-4o")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    
    text_summaries = []
    table_summaries = []
    
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    elif texts:
        text_summaries = texts
        
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
    
    return text_summaries, table_summaries

def create_multi_vector_retriever(vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images):
    """Create retriever that indexes summaries but returns raw content"""
    store = InMemoryStore()
    id_key = "doc_id"
    
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    
    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))
    
    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    if image_summaries:
        add_documents(retriever, image_summaries, images)
    
    return retriever

def split_image_text_types(docs):
    """Split base64-encoded images and texts"""
    b64_images = []
    texts = []
    for doc in docs:
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}

def looks_like_base64(sb):
    """Check if string looks like base64"""
    return bool(re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb))

def is_image_data(b64data):
    """Check if base64 data is an image"""
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]
        return any(header.startswith(sig) for sig in image_signatures)
    except Exception:
        return False

def resize_base64_image(base64_string, size=(128, 128)):
    """Resize base64 encoded image"""
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    resized_img = img.resize(size, Image.LANCZOS)
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def img_prompt_func(data_dict):
    """Create prompt for GPT-4o"""
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []
    
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            messages.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"}
            })
    
    messages.append({
        "type": "text",
        "text": (
            "You are insurance expert analyzing policy documents.\n"
            "You will be given a mix of text, tables, and image(s) usually of charts or graphs.\n"
            "Use this information to provide concise and precise answers to the user question. \n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        )
    })
    
    return [HumanMessage(content=messages)]

def create_multimodal_rag_chain(retriever):
    """Create multimodal RAG chain"""
    model = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=5000)
    
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )
    
    return chain

# Streamlit UI
st.title("Multimodal RAG System")
st.write("Upload a PDF document to analyze with GPT-4o")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None and not st.session_state.processed_files:
    with st.spinner("Processing document..."):
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file
            temp_pdf_path = os.path.join(temp_dir, "uploaded.pdf")
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Extract elements
            raw_pdf_elements = extract_pdf_elements(temp_pdf_path, temp_dir)
            texts, tables = categorize_elements(raw_pdf_elements)
            
            # Process texts
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=4000, chunk_overlap=0
            )
            joined_texts = " ".join(texts)
            texts_4k_token = text_splitter.split_text(joined_texts)
            
            # Generate summaries
            text_summaries, table_summaries = generate_text_summaries(
                texts_4k_token, tables, summarize_texts=True
            )
            
            # Process images
            img_base64_list = []
            image_summaries = []
            prompt = """You are an assistant tasked with summarizing images for retrieval. \
            These summaries will be embedded and used to retrieve the raw image. \
            Give a concise summary of the image that is well optimized for retrieval."""
            
            for img_file in sorted(os.listdir(temp_dir)):
                if img_file.endswith(".jpg"):
                    img_path = os.path.join(temp_dir, img_file)
                    base64_image = encode_image(img_path)
                    img_base64_list.append(base64_image)
                    image_summaries.append(image_summarize(base64_image, prompt))
            
            # Create retriever
            vectorstore = Chroma(
                collection_name="mm_rag_streamlit",
                embedding_function=OpenAIEmbeddings()
            )
            
            st.session_state.retriever = create_multi_vector_retriever(
                vectorstore,
                text_summaries,
                texts,
                table_summaries,
                tables,
                image_summaries,
                img_base64_list
            )
            
            st.session_state.processed_files = True
            st.success("Document processed successfully!")

# Query input
if st.session_state.processed_files:
    query = st.text_input("Enter your question:")
    
    if query:
        with st.spinner("Analyzing..."):
            # Create and run RAG chain
            chain = create_multimodal_rag_chain(st.session_state.retriever)
            response = chain.invoke(query)
            
            # Display response
            st.write("### Analysis")
            st.write(response)
            
            # Show retrieved documents
            st.write("### Retrieved Documents")
            docs = st.session_state.retriever.invoke(query)
            
            for i, doc in enumerate(docs):
                st.write(f"Document {i + 1}:")
                if looks_like_base64(doc) and is_image_data(doc):
                    st.image(f"data:image/jpeg;base64,{doc}")
                else:
                    st.write(doc)

# Reset button
if st.session_state.processed_files:
    if st.button("Process New Document"):
        st.session_state.processed_files = False
        st.session_state.retriever = None
        st.experimental_rerun()