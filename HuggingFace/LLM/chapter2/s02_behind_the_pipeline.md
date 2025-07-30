## Steps within pipeline()  
![Steps within pipeline](../../../pics/Steps%20of%20pipeline.JPG)
### 1. Preprocessing  
Like other NN, transformer models can not process raw text directly, so the first step of our pipeline is to convert the text inputs into numbers that the model can make sense of. To do this we use a **tokenizer**, which will be responsible for:  
    - Splitting the input into words, subwords, or symbols (like punctuation) that are called **tokens**  
    - Mapping each token to an **integer (input_ids)**  
    - Adding **additional inputs** that may be useful to the model, e.g. **attention_masks**  
```
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
# padding is used to make all input sentences the same length, as tensors need to be rectangular
# truncation is used to truncate sentenses that are longer than the model can handle
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

>> {
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}
```
### 2. Passing the inputs through the model  -- Inference
The values we get as output from our model don’t necessarily make sense by themselves. Those are not probabilities but **logits**, the raw, unnormalized scores outputted by the last layer of the model.
```
# There are many different architectures available designed to tackle a specific task, for example, AutoModel, BertModel, AutoModelForCausalLM etc.
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(output.logits)

>> tensor([[-1.5607,  1.6123],
           [ 4.1692, -3.3464]], grad_fn=<AddmmBackward>)
```
### 3. Postprocessing  
To convert logits to **probabilities**, they need to go through a **SoftMax** layer:  
```
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

>> tensor([[4.0195e-02, 9.5980e-01],
           [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward>)
```  
To get the labels corresponding to each position, we can inspect the id2label attribute of the model config:  
```
model.config.id2label

>> {0: 'NEGATIVE', 1: 'POSITIVE'}
```  
Now we can conclude that the model predicted the following:  
First sentence: NEGATIVE: 0.0402, POSITIVE: 0.9598  
Second sentence: NEGATIVE: 0.9995, POSITIVE: 0.0005