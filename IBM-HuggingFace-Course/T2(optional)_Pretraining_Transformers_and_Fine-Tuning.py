from transformers import pipeline
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoConfig,AutoModelForCausalLM,AutoModelForSequenceClassification,BertConfig,BertForMaskedLM,TrainingArguments, Trainer, TrainingArguments
from transformers import AutoTokenizer,BertTokenizerFast,TextDataset,DataCollatorForLanguageModeling
from datasets import load_dataset
from tqdm.auto import tqdm
import math
import time
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Pretraining and self-supervised fine-tuning
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(pipe("This moive was really")[0]["generated_text"])

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
print(dataset)