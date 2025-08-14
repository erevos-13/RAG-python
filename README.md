# IBS RAG Assistant 🤖📚

A Retrieval-Augmented Generation (RAG) system built with FastAPI and LangChain to provide intelligent answers by leveraging uploaded PDF documents and OpenAI's GPT models.

## 🎯 What is RAG (Retrieval-Augmented Generation)?

RAG is a cutting-edge AI technique that combines the power of **retrieval** and **generation** to create more accurate, contextual, and up-to-date AI responses:

### How RAG Works:
1. **Document Ingestion**: PDF documents are uploaded and processed
2. **Text Chunking**: Documents are split into manageable chunks
3. **Embedding Creation**: Text chunks are converted to vector embeddings using OpenAI's embedding models
4. **Vector Storage**: Embeddings are stored in a ChromaDB vector database
5. **Similarity Search**: When a question is asked, the system finds the most relevant document chunks
6. **Context-Aware Generation**: The retrieved context is fed to a language model (GPT-4) to generate accurate answers

### Why RAG is Revolutionary:
- **Accuracy**: Answers are grounded in your specific documents, reducing hallucinations
- **Currency**: Always up-to-date with your latest documents
- **Transparency**: You know exactly which documents informed the answer
- **Customization**: Tailored to your specific domain knowledge

## 🚀 Features

- **📄 PDF Upload**: Upload medical documents, research papers, and IBS-related content
- **🔍 Intelligent Retrieval**: Uses MMR (Maximal Marginal Relevance) for diverse, relevant results
- **💬 Contextual Q&A**: Ask questions and get answers based on your uploaded documents
- **⚡ Fast API**: RESTful API built with FastAPI for high performance
- **🧠 Advanced AI**: Powered by OpenAI's GPT-4 and text-embedding-ada-002

## 🐍 Why Python is Optimal for RAG Systems

Python has emerged as the **dominant language** for AI and RAG applications, and here's why it outperforms other programming languages:

### 1. **Rich AI/ML Ecosystem** 🔬
```python
# Extensive libraries available out-of-the-box
from langchain import *          # LLM orchestration
from chromadb import *           # Vector databases
from transformers import *       # Hugging Face models
from openai import *            # OpenAI integration
```

### 2. **Rapid Prototyping** ⚡
- **Interpreted Language**: No compilation step - immediate feedback
- **Dynamic Typing**: Quick iterations and experimentation
- **REPL Environment**: Interactive development with Jupyter notebooks

### 3. **Data Science Integration** 📊
```python
import pandas as pd              # Data manipulation
import numpy as np               # Numerical computing
import matplotlib.pyplot as plt  # Visualization
import scikit-learn as sklearn   # Traditional ML
```

### 4. **Compared to Other Languages**:

| Language | AI Libraries | Learning Curve | Community | Performance* |
|----------|-------------|---------------|-----------|-------------|
| **Python** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| JavaScript | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Java | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| C++ | ⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| R | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

*Performance for AI workloads with proper optimization

### 5. **Industry Standard** 🏭
- **Research**: Most AI research papers provide Python implementations
- **Production**: Companies like OpenAI, Google, Meta use Python for AI systems
- **Education**: Universities teach AI/ML primarily through Python

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- OpenAI API Key

### 1. Clone the Repository
```bash
git clone https://github.com/erevos-13/RAG-python.git
cd ibs-rag
```

### 2. Create Virtual Environment
```bash
python -m venv rag-open-ai
source rag-open-ai/bin/activate  # On Windows: rrag-open-ai\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install fastapi uvicorn langchain langchain-openai langchain-community
pip install chromadb pypdf python-dotenv python-multipart
```

### 4. Environment Configuration
Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run the Application
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

### Interactive Docs
Visit `http://localhost:8000/docs` for Swagger UI documentation.

### Endpoints

#### 1. Upload Document
```http
POST /upload
Content-Type: multipart/form-data

Body: file (PDF)
```

**Example:**
```bash
curl -X POST "http://localhost:8000/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@ibs_research.pdf"
```

#### 2. Ask Question
```http
GET /ask?q=your_question_here
```

**Example:**
```bash
curl -X GET "http://localhost:8000/ask?q=What do you know about orfas?"
```

**Response:**
```json
{
  "question": "What do you know about orfas?",
  "answer": "Based on the uploaded documents, orfeas is ..."
}
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Upload    │───▶│  Text Chunking  │───▶│   Embeddings    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   Retrieval     │◀───│   ChromaDB      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │
        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐
│   GPT-4 Model   │◀───│   Context +     │
│                 │    │   Question      │
└─────────────────┘    └─────────────────┘
        │
        ▼
┌─────────────────┐
│   AI Response   │
└─────────────────┘
```

## 🔧 Technical Components

### Document Processing
- **PyPDFLoader**: Extracts text from PDF files
- **CharacterTextSplitter**: Splits documents into chunks (500 chars, 50 overlap)
- **Text Normalization**: Removes extra whitespace for cleaner processing

### Vector Storage
- **ChromaDB**: Persistent vector database
- **OpenAI Embeddings**: text-embedding-ada-002 model
- **MMR Retrieval**: Maximal Marginal Relevance for diverse results

### Language Model
- **GPT-4**: Primary language model for answer generation
- **Seed Parameter**: Ensures reproducible outputs (seed=365)
- **Token Limit**: 250 tokens for concise responses

## 🚀 Usage Examples

### Generic Research Assistant
```python
# Upload medical research papers
transcriptorpheusandeurydicePOST /upload -> "ibs_treatment_guidelines.pdf"

# Ask specific questions
GET /ask?q="How is Orfeas?"
```



## 🛡️ Security Considerations

- **API Keys**: Store OpenAI API keys in environment variables
- **File Validation**: Implement proper file type validation
- **Rate Limiting**: Consider implementing rate limiting for production
- **Data Privacy**: Ensure HIPAA compliance for medical data

## 📈 Performance Optimization

### Current Configuration
- **Chunk Size**: 500 characters (optimal for large text)
- **Chunk Overlap**: 50 characters (maintains context)
- **Retrieval**: Top 3 results with MMR (λ=0.7)
- **Model**: GPT-4 with controlled output length

### Scaling Recommendations
- **Async Processing**: Use FastAPI's async capabilities for large uploads
- **Caching**: Implement Redis for frequently asked questions
- **Load Balancing**: Deploy multiple instances for high traffic

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: For the incredible RAG framework
- **OpenAI**: For powerful language models and embeddings
- **ChromaDB**: For efficient vector storage
- **FastAPI**: For the high-performance web framework

## 📞 Support

For questions, issues, or contributions:
- 📧 Email: [orfeas@voutsaridiso.com]
- 🐛 Issues: [GitHub Issues](https://github.com/erevos-13/RAG-python/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/erevos-13/RAG-python/discussions)

---

**Built with ❤️ and Python for better research through AI**
