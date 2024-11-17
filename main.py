import gradio as gr
import os
import dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate


dotenv.load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initializing the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
print("Groq model initialized.")

# Initialize HuggingFace embeddings
print("Initializing HuggingFace Embeddings...")
embedding_function = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={"normalize_embeddings": True}
)
print("Embeddings initialized.")

# Predefined responses for greetings and specific queries
predefined_responses = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hi there! How can I help?",
    "how are you": "I'm just a virtual assistant, but I'm here and ready to help!",
    "thank you": "You're welcome! Let me know if there's anything else you need.",
    "thanks": "You're welcome! Have a great day!",
    "bye": "Goodbye! Take care.",
    "who made you": "I was created by Sachin Raj M. He made this in one day and shipped it for greater good.",
    "who created you": "Sachin Raj M is my creator. He built this system for a better, more efficient way to navigate through PDF documents.",
    "creator of pdfchatbot": "Sachin Raj M. He crafted this chatbot in one day to help users quickly find answers in PDFs.",
    "when was you created": "I was created on 17 November 2024.",
    "why was you created": (
        "I was created to address the problem of endless scrolling through PDF pages. "
        "Many people struggled to find information quickly, so this app was designed to make searching within PDFs fast and efficient."
    )
}

# Function to load and add documents


def add_docs(path):
    print("Loading documents from:", path)
    try:
        loader = PyPDFLoader(file_path=path)
        docs = loader.load_and_split(
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=300,
                length_function=len,
                is_separator_regex=False
            )
        )
        print(f"Loaded {len(docs)} document chunks.")

        # Ensure persistence directory exists
        persist_dir = "output/general_knowledge"
        os.makedirs(persist_dir, exist_ok=True)

        # Initialize Chroma database and add documents
        model_vectorstore = Chroma
        db = model_vectorstore.from_documents(
            documents=docs,
            embedding=embedding_function,
            persist_directory=persist_dir
        )
        print("Documents added to vector store.")
        return db
    except Exception as e:
        print("Error in add_docs:", e)
        return None

# Function to answer a query based on stored documents


def answer_query(message, chat_history):
    print("Received query:", message)

    # Check for predefined responses
    lower_message = message.lower().strip()
    if lower_message in predefined_responses:
        response_content = predefined_responses[lower_message]
        formatted_response = {
            "role": "assistant",
            "content": response_content
        }

        # Update chat history
        chat_history.append({"role": "user", "content": message})
        chat_history.append(formatted_response)
        print("Predefined response generated:", response_content)
        return "", chat_history

    # If not a predefined response, proceed with RAG pipeline
    try:
        # Initialize compressor and retrievers
        base_compressor = LLMChainExtractor.from_llm(llm)
        db = Chroma(
            persist_directory="output/general_knowledge",
            embedding_function=embedding_function
        )
        base_retriever = db.as_retriever()
        mq_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever, llm=llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=base_compressor,
            base_retriever=mq_retriever
        )

        # Retrieve relevant documents
        matched_docs = compression_retriever.invoke(input=message)
        print(f"Found {len(matched_docs)} relevant documents.")

        # Concatenate document content
        context = "\n\n".join([doc.page_content for doc in matched_docs])

        # Define the prompt template
        template = """
        You are a knowledgeable assistant. Use the context provided to answer the question in detail and with clarity. 
        Structure your response into clear paragraphs and provide additional insights if they are relevant, but do not use information outside the context.
        If the given context does not have the information, respond: 'I don't know the answer to this question.'
        Context: ```{context}```
        ----------------------------
        Question: {query}
        ----------------------------
        Answer: """

        # Generate prompt
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            template=template)
        chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
        prompt = chat_prompt.format_prompt(query=message, context=context)

        # Get response from the language model
        response = llm.invoke(input=prompt.to_messages())

        # Ensure the response is in the correct format
        formatted_response = {
            "role": "assistant",
            "content": response.content
        }

        # Update chat history
        chat_history.append({"role": "user", "content": message})
        chat_history.append(formatted_response)
        print("Response generated:", response.content)
        return "", chat_history
    except Exception as e:
        print("Error in answer_query:", e)
        return "", chat_history


# Build Gradio interface
with gr.Blocks() as demo:
    gr.HTML("<h1 align='center'>RapiDoc</h1>")

    with gr.Row():
        upload_files = gr.File(label='Upload a PDF', file_types=[
                               '.pdf'], file_count='single')

    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(label="Enter your question here")
    upload_files.upload(add_docs, upload_files)
    msg.submit(answer_query, [msg, chatbot], [msg, chatbot])

# Start the app
print("Launching Gradio app...")
demo.launch()
print("App is running...")
