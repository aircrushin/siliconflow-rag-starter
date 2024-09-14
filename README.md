# SiliconFlow RAG Project

This project implements a Retrieval-Augmented Generation (RAG) system using the SiliconFlow API and LangChain framework. It demonstrates how to create a question-answering system that retrieves relevant information from a document and generates responses using a large language model.

## Features

- Document loading and splitting
- Text embedding using Hugging Face models
- Vector storage with Chroma
- Custom LLM integration with SiliconFlow API
- RAG chain implementation for question answering

## Requirements

To run this project, you need to install the following dependencies:


## Setup

1. Clone the repository
2. Install the required packages: `pip install -r requirements.txt`
3. Create a `.env` file in the project root and add your SiliconFlow API key:
   ```bash
   OPENAI_API_KEY=your_siliconflow_api_key_here
   ```

## Usage

The main script is `rag.py`. It performs the following steps:

1. Loads a document from `base.txt`
2. Splits the document into chunks
3. Creates embeddings and stores them in a Chroma vector store
4. Sets up a custom SiliconFlow LLM
5. Creates a RAG chain
6. Answers a sample question

To run the script:

```
python rag.py
```

## Customization

You can customize the following parameters in `rag.py`:

- Text splitting parameters (chunk size, overlap)
- Embedding model
- Number of retrieved documents (k)
- SiliconFlow model name

## License

[MIT License](https://opensource.org/licenses/MIT)