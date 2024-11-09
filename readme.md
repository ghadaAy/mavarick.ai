# Project Overview

This project combines a hybrid search architecture using the Milvus vector database with multiple open-source models to create an efficient, lightweight document retrieval and question-answering system.
## Pre-requisites:
### Docker & Docker compose
Ensure Docker and the docker compose plugin are installed on your system to build and run the containers ([guide](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)).

### NVIDIA container-toolkit

Ensure that the NVIDIA container toolkit is installed on the host system to enable the container to access the GPU. Without it, startup and inference will be very slow ([guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker)).

## How to Run
To build and run the project, follow these steps:
**Note: If Ollama cannot access the GPU, performance will be significantly slower.**

1. Start the containers
```bash
docker compose up --build -d
```
   the backend will take a while to start as indexing and model loading and downloading happen at startup. This is only viable in a testing setup, in a real world scenario indexing would occur on demand in a dedicated endpoint.
   
2. Please wait until the container has started
   
4. Go to your browser at **http://localhost:8000/api/v1/docs**
   
6. click the **get_answer** endpoint
   
8. click **try it out**
   
10. input your query in the text box and click **execute**
    
12. wait for the answer to appear below


## Key Technologies

- **Vector Database: Milvus**
  This project uses Milvus with hybrid search capabilities, allowing efficient storage and retrieval of high-dimensional embeddings.

- **Language Model: Gemma 2B**
  We selected the Gemma 2B open-source language model for its speed and compact size, optimizing performance for rapid queries and lightweight processing.

- **Embedding Model: Nomic-Embed-Text**
  Nomic-Embed-Text is used for generating embeddings due to its quick processing and minimal memory requirements, ensuring efficient document retrieval from Milvus.

- **File Splitting: LLM Sherpa**
  LLM Sherpa is utilized to parse PDFs, preserving hierarchical layout information for contextually accurate splitting in structured documents.
  - **Fallback:** If LLM Sherpa fails, we use the **RecursiveCharacterTextSplitter** for generic text. This splitter attempts to maintain semantic coherence by splitting text based on a prioritized list of characters (`["\n\n", "\n", " ", ""]`), aiming to keep paragraphs, sentences, and words intact.

- **Query Handling Platform: Ollama**
  Ollama is used in development for its simplicity. For production deployment, either **VLLM** ([GitHub link](https://github.com/vllm-project/vllm)) or **TensorRT** ([GitHub link](https://github.com/NVIDIA/TensorRT)) should be used for better scalability and performance.

## Architecture

### Step 1: Document Preparation
1. **Split Documents**
   Documents are initially split using **LLM Sherpa**. If unsuccessful, **RecursiveCharacterTextSplitter** is used as a fallback.

### Step 2: Query Processing
1. **Contextualize Query**
   Each user query is rephrased to be self-contained, using historical context when needed to ensure the language model fully understands without requiring the entire context.

1. **Retrieve Documents**
   Relevant documents are retrieved from the Milvus vector database based on the query.

1. **Grade Retrieved Documents**
   A grading language model evaluates retrieved documents, filtering out irrelevant results to improve the accuracy of the response.

1. **Generate Answer**
   The graded documents, along with the contextualized query, are passed to a generator language model to produce an answer.

1. **Check for Hallucinations**
   If the generated answer appears to contain hallucinations, the model re-generates a response strictly limited to the retrieved documents.

### Step 3: Enhanced Query Processing
1. **Keyword Extraction**
   Keywords from the user’s question are extracted to refine document retrieval further.

2. **Repeat Retrieval and Answer Generation**
   The previous steps (document retrieval, grading, answer generation, and hallucination check) are repeated to refine the response based on extracted keywords.

---

This setup is designed to ensure accurate responses by leveraging lightweight, high-performance models in a structured retrieval and generation workflow.
To see the graph for the process please navigate to docs folder.

## Next Steps for a Production-Ready RAG System

To build a more robust and production-ready retrieval-augmented generation (RAG) system, the following enhancements are recommended:

1. **Use Larger, Faster Models**
   Implement models such as **LLAMA 3.1 (8B or 13B)** to improve answer quality and depth.

2. **Advanced Embedding Models**
   Integrate embedding models ranked at the top of industry leaderboards for higher retrieval accuracy.

3. **Graph Knowledge Bases**
   Use graph-based knowledge bases to enhance the understanding of relationships between entities, providing more contextually accurate answers. A promising tool for this is **LightRAG** ([GitHub link](https://github.com/HKUDS/LightRAG)), which is designed to be both fast and cost-effective. Note: Testing and debugging are needed to assess integration with this tool.

4. **Enhanced Question Summarization with Raptor**
   Implement **Raptor** ([GitHub link](https://github.com/parthsarthi03/raptor)) for hierarchical chunk summarization, enabling the system to handle complex, high-level questions more effectively.

5. **Answer Streaming**
   Enable answer streaming to improve user experience by allowing responses to be displayed progressively as they are generated.

6. **Endpoint for File Upload to Knowledge Base**
    Implement an endpoint that enables users or team members to upload files, automatically adding them to the knowledge base to support answering questions related to their content.
---

This setup and roadmap aim to ensure efficient and accurate responses while providing a pathway to scale and enhance the system for production environments.

## Disclaimer
- This project uses includes` app.telemetry` module for structured logging (`structlog`) and tracing (`OpenTelemetry`), which is a logging template I reuse across multiple projects.
- The module's code is based on publicly available gists and examples and was not custom-built for this project.
