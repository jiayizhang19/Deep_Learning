import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline
import gc

# ===================================
# Text classification with DistilBERT
# ===================================

# Step 1: Load the model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Step 2: Preprocess the input text
text = "Congratulations! You've won a free ticket to the Bahamas. Reply WIN to claim."
inputs = tokenizer(text, return_tensors="pt")
# The input_ids and attention mask in inputs are numerical representations of the input text, as BERT-based models can't understand raw text.
print(inputs)

# Step 3: Perform inference
with torch.no_grad():
    outputs = model(**inputs)
# The logit is just a number, which means the score of the two classes. These values are not yet normalized with a sum of 1.
logits = outputs.logits

# Step 4: Post-processing the output
# Convert logits to probabilities, softmax normalizes the logits into probabilities (summing to 1)
probs = torch.softmax(logits, dim=-1)
# Get the predicted class, argmax returns the index of the highest score
predicted_class = torch.argmax(probs, dim=-1)
# Map the predicted class to the label, with reference to the model designed
labels = ["NEGATIVE", "POSITIVE"]
print(labels[predicted_class])


# ==========================
# Text generation with GPT-2
# ==========================
# Step 1: Load the model and tokenizer 
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Step 2: Preprocess the input text
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
inputs

# Step 3: Perform inference
output_ids = model.generate(
    inputs.input_ids, 
    attention_mask=inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    max_length=50, 
    num_return_sequences=1
)
output_ids
# Or use the below with the same purpose
# with torch.no_grad():
#     outputs = model(**inputs) 
# outputs

# Step 4: Post-processing the output
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_text)


# ================================================================
# Text classification using pipeline()
# https://huggingface.co/docs/transformers/main_classes/pipelines
# ================================================================
# transformers.pipeline(
#     task: str,
#     model: Optional = None,
#     config: Optional = None,
#     tokenizer: Optional = None,
#     feature_extractor: Optional = None,
#     framework: Optional = None,
#     revision: str = 'main',
#     use_fast: bool = True,
#     model_kwargs: Dict[str, Any] = None,
#     **kwargs
# )

# Example 1: Text classification
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("Congratulations! You've won a free ticket to the Bahamas. Reply WIN to claim.")
print(result)

# Example 2: Language detection
classifier = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
result = classifier("你好，这是一段中文文本")
print(result)

# Example 3: Text generation 
generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time", max_length=50, num_return_sequences=1, truncation=True)
print(result)

# Example 4: Text generation using T5
generator = pipeline("text2text-generation", model="google/mt5-small")
prompt = "translate English to Mandarin: how are you?"
result = generator(prompt, max_length=50, num_return_sequences=1)
print(result[0]["generated_text"])

# Example 5: Fill mask
filler = pipeline("fill-mask", "bert-base-uncased")
prompt = "I was born in China, so I am a [MASK]."
result = filler(prompt)
print(result)

# Release resources
del tokenizer, model, classifier, generator, filler
gc.collect()