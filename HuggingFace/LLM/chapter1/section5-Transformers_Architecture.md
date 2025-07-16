### Architecture of transformers / Types of language models
- **Encoder-only models** (like BERT) for text classification tasks
    - These models use a bidirectional approach to understand context from both directions. 
    - They are best suited for tasks that require *deep understanding of text*, such as classification, named entity recognition, and question answering.
- **Decoder-only models** (like GPT, Llama) for text generation tasks
    - These models process text from left to right and are particularly good at *text generation tasks*.
    - They can complete sentences, write essays, or even generate code based on a prompt.
- **Encoder-decoder models** (like T5, BART)
    - These models combine both approaches, using an encoder to udnerstand the input and decoder to generate output. 
    - They excel at *sequence-to-sequence tasks* like translation, summarization, and question answering.