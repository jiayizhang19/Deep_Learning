### Transformer Architecture / Types of Language Models
- **Encoder-only models** (auto-encoding models)
    - These models are often characterized as having **bi-directional** attention. The attention layers can access **all the words** in the initial sentence. 
    - They are best suited for tasks requiring an understanding of the full sentense, such as 
        - sentence classification 
        - named entity recognition
        - question answering
    - Examples: BERT, DistilBERT, ModernBERT
- **Decoder-only models** (auto-regressive models)
    - The attention layers can only access the words positioned before it in the sentence. 
    - These models process text from left to right and are particularly good at *text generation tasks*, such as 
        - text generation / code generation
        - question answering
    - Examples: Modern Large Language Models (LLM), like Llama, GPT, DeekSeek
- **Encoder-decoder models** (sequence-to-sequence models)
    - These models combine both approaches, using an encoder to udnerstand the input and decoder to generate output. 
    - They excel at *sequence-to-sequence tasks*, such as
        - machine translation
        - text summarization
        - grammar correction 
        - question answering
    - Examples: T5, BART