import torch
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel

def bert_model_and_tokenizer():
    MODEL_NAME = "distilbert-base-uncased"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    return device, tokenizer, model
