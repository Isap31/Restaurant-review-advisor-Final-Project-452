# Restaurant-review-advisor-Final-Project-452
Final Project (DEMO)

AI-powered system for restaurant review sentiment analysis and knowledge retrieval.

## Project Overview

This application combines fine-tuned sentiment analysis with Retrieval-Augmented Generation (RAG) to analyze restaurant reviews and answer questions based on real customer feedback.

**Course:** INFO 452 - Fall 2025  
**Student:** Caitlin Przywara  
**Submission Date:** December 1, 2025

## System Architecture

### Component 1: Fine-Tuned Sentiment Model
- **Model:** DistilBERT (distilbert-base-uncased)
- **Dataset:** Yelp Review Polarity (fancyzhx/yelp_polarity)
- **Training Data:** 50,000 restaurant reviews
- **Test Data:** 10,000 restaurant reviews
- **Method:** Transfer learning (fine-tuned for 3 epochs)
- **Performance:** 94.93% accuracy
- **Task:** Binary sentiment classification (Positive/Negative)

### Component 2: RAG System
- **Retrieval:** ChromaDB vector database with 5,000 Yelp reviews
- **Augmentation:** Context injection from retrieved reviews
- **Generation:** FLAN-T5-Small language model
- **Documents:** Real customer reviews from Hugging Face dataset
- **Purpose:** Answer questions about restaurants using actual customer feedback

### Component 3: Gradio GUI
- **Framework:** Gradio
- **Features:**
  - Sentiment Analysis mode
  - Knowledge Query (RAG) mode
  - Complete Analysis mode (combines both)
- **Deployment:** Hugging Face Spaces

## How to Run

### Prerequisites
```bash
Python 3.8+
Google Colab (recommended)
GPU recommended for training (T4 or better)
```

### Installation
```bash
pip install transformers datasets accelerate chromadb sentence-transformers gradio torch
```

### Running the Project

**1. Train Sentiment Model (10 minutes):**
```python
# Run cells in section: "Component 1 - Sentiment Analysis Training"
# Model will be saved to: restaurant_sentiment_model/
```

**2. Setup RAG System (10-15 minutes):**
```python
# Run cells in section: "Component 2 - RAG System Setup"
# Vector database will be created with 5,000 reviews
```

**3. Launch Gradio App:**
```python
# Run cells in section: "Component 3 - Gradio Interface"
# A public share link will be generated
```

## Results

### Sentiment Model Performance
- **Training Accuracy:** Stayed at 94.93% from (Epoch 1) to (Epoch 3)
- **Test Accuracy:** 94.93%
- **Confidence Scores:** Average 99%+ on test examples

### RAG System Performance
- **Retrieval:** Successfully finds relevant reviews based on semantic similarity
- **Generation:** Produces natural language answers based on customer feedback
- **Response Time:** ~2-3 seconds per query

**Hugging Face URL:** https://huggingface.co/Isap31/restaurant-sentiment-distilbert

## Live Demo

**Hugging Face Space:** https://huggingface.co/spaces/Isap31/Restaurant-review-advisor-Final-Project-452

**Try these examples:**
- Sentiment: "This restaurant exceeded all expectations!"
- RAG Query: "What do customers say about food quality?"
- Complete Analysis: Combines both for comprehensive insights

## Project Structure

```
restaurant-review-advisor/
├── FinalProject.ipynb          # Main Jupyter notebook with all code
├── README.md                    # This file
├── reflection.pdf               # One-page reflection paper
├── restaurant_sentiment_model/  # Saved fine-tuned model (generated)
└── requirements.txt             # Python dependencies
```

## AI Assistance Documentation:

This project was developed with assistance from my favorite GPT - Claude AI (Anthropic). Claude helped me with:

### Code Development:
- Debugging CUDA errors during model training
- Structuring the RAG pipeline (Retrieval → Augmentation → Generation)
- Creating Gradio interface components
- Dataset selection and preprocessing

### Problem Solving:
- Identifying label mismatch issues in initial dataset
- Switching from cryptocurrency to restaurant review domain
- Understanding the difference between simple retrieval and true RAG
- Optimizing model training parameters

### Documentation:
- Writing clear code comments and explanations
- Structuring README and technical documentation
- Preparing presentation talking points

**Student Contributions:**
- All project decisions and direction
- Running all training and testing
- Understanding and explaining technical concepts
- Integration and deployment
- Presentation and documentation review

## Technologies Used:

| Component | Technology | Version/Model ID |
|-----------|-----------|------------------|
| Sentiment Model | DistilBERT | distilbert-base-uncased |
| Dataset | Yelp Review Polarity | fancyzhx/yelp_polarity |
| RAG Generation | FLAN-T5 | google/flan-t5-small |
| Vector Database | ChromaDB | Latest |
| Embeddings | Sentence Transformers | all-MiniLM-L6-v2 |
| GUI Framework | Gradio | Latest |
| Training Framework | Hugging Face Transformers | Latest |

## Key Learnings:

### Technical Insights:
1. **Dataset Quality > Dataset Size:** A clean, well-labeled dataset (Yelp) outperforms larger noisy datasets
2. **Transfer Learning Works:** Fine-tuning pre-trained models achieves high accuracy with modest training data
3. **RAG Architecture:** True RAG requires all three components: Retrieval, Augmentation, and Generation

### Challenges that I overcame:
1. Initial dataset (Bitcoin tweets) had class imbalance and label issues
2. CUDA errors from model/label mismatch required debugging and dataset switch
3. Understanding difference between simple document retrieval and true RAG with generation

### Project Management:
1. Switching datasets mid-project was the right decision (63% → 94.93% accuracy)
2. Using same dataset for both components created strong project cohesion
3. Incremental testing and validation prevented larger issues

## Future Improvements

1. **Multi-Restaurant Analysis:** Extend to analyze specific restaurants by name
2. **Trending Insights:** Identify emerging themes in recent reviews
3. **Aspect-Based Sentiment:** Break down sentiment by category (food, service, ambiance)
4. **Real-Time Data:** Connect to live Yelp API for current reviews
5. **Multilingual Support:** Extend to non-English reviews

## License

This project is submitted as coursework for INFO 452 at Virginia Commonwealth University

## Contact Information:

**Student:** Caitlin Przywara 
**Email:** przywarac@vcu.edu 
**GitHub:** Isap31 
**Project:** INFO 452 Final Project - Fall 2025

**Last Updated:** November 29, 2025
