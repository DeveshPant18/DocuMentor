# DocuMentor 🎓

**Chat with your documents using the power of local LLMs and Retrieval-Augmented Generation (RAG).**

DocuMentor is an intelligent study assistant that transforms your static documents into a dynamic conversational partner. Upload your course materials, research papers, or notes, and get instant, context-aware answers without your data ever leaving your machine.

## ✨ Features

* **Interactive Chat Interface:** A user-friendly interface built with Streamlit for seamless interaction.
* **Dynamic Document Upload:** Upload one or more PDF files directly through the web UI.
* **Advanced RAG Pipeline:** Utilizes a **ParentDocumentRetriever** to provide LLMs with better context, leading to more accurate and comprehensive answers.
* **Conversational Memory:** The assistant remembers the context of your conversation, allowing for natural follow-up questions.
* **100% Local & Private:** Runs entirely on your machine using **Ollama** and Llama 3. Your documents and conversations are never sent to the cloud.
* **Easy Session Management:** A "Clear Conversation" button to instantly reset the chat and document context.

## 🛠️ Tech Stack

* **Framework:** [Streamlit](https://streamlit.io/)
* **LLM Orchestration:** [LangChain](https://www.langchain.com/)
* **Large Language Model (LLM):** [Llama 3](https://ollama.com/library/llama3) (or any model served by Ollama)
* **Vector Store:** [ChromaDB](https://www.trychroma.com/)
* **Embeddings:** [Hugging Face Sentence-Transformers](https://huggingface.co/sentence-transformers)

## 🚀 Getting Started

Follow these instructions to set up and run DocuMentor on your local machine.

### Prerequisites

* Python 3.9+
* [Ollama](https://ollama.com/) installed and running.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DeveshPant18/DocuMentor.git
    cd DocuMentor
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Pull the Llama 3 model via Ollama:**
    *Make sure the Ollama application is running before executing this command.*
    ```bash
    ollama pull llama3
    ```

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    Your web browser should automatically open to the application.

2.  **Upload Your Documents:**
    * Use the file uploader in the sidebar to select one or more PDF files.

3.  **Process Documents:**
    * Click the **"Process Documents"** button.
    * Wait for the progress bar to complete. The app will load the files, split them into chunks, and create a vector index.

4.  **Start Chatting:**
    * Once the documents are processed, you can ask questions about their content in the chat input box.

5.  **Clear the Conversation:**
    * Click the **"Clear Conversation"** button in the sidebar to start over with new documents.

## 🔮 Future Improvements

This project has a solid foundation, but there's always room to grow. Here are some potential next steps:

* **Enhanced Document Support:** Add support for more file types (`.docx`, `.txt`, `.pptx`) and improve the parsing of complex PDFs with tables and images using libraries like `unstructured`.
* **Response Streaming:** Implement token streaming from the LLM to make the application feel more responsive.
* **Performance Optimization:** Introduce caching for LLM responses to handle repeated questions instantly.
* **Containerization:** Package the application with Docker for easier deployment and dependency management.