import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

def download_models():
    print("Downloading SentenceTransformer model...")
    SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Downloading DistilBERT tokenizer and model...")
    # Replace 'distilbert-base-uncased' with your specific fine-tuned model path if hosted
    # Or just standard DistilBERT if we don't have a fine-tuned one deployed yet
    AutoTokenizer.from_pretrained("distilbert-base-uncased")
    # For safety classifier, we might not want to download the base model if we expect a fine-tuned local one,
    # but the instructions say to cache models.
    # AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

if __name__ == "__main__":
    download_models()
