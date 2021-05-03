# Deep Learning School Part 2: Natural Language Processing with Deep Learning

This repository shows my first dive into NLP. Huge thanks
to [Deep Learning School](https://en.dlschool.org/) organised with the help of
[Phystech School of Applied Mathematics and Informatics](https://mipt.ru/english/edu/phystechschools/psami) (Факультет Прикладной Математики и Информатики МФТИ).

## Syllabus

Content from Spring 2021: https://stepik.org/course/92488

1. Organisational information
2. [Intro to NLP](https://github.com/GeorgeBatch/nlp_from_dls/tree/main/02_Intro_to_NLP)
    1. Lecture
        * NLP tasks and typical pipeline
        * One-Hot Encoding, Bag of words, TF-IDF, Colocations and n-grams, pointwise mutual information
        * Context embeddings using SVD
        * Text classification and regression
    2. Seminar
        * tokenization, stop words
        * stemming, lemmatization
        * TF-IDF, cosine similarity
        * Text classification using all the above
        * NLP Libraries: nltk, [razdel](https://github.com/natasha/razdel), pymorphy2, spacy, rnnmorph 
3. [Coursework: Simple Embeddings](https://github.com/GeorgeBatch/nlp_from_dls/tree/main/03_HW_Simple_embeddings)
    * todo: add the description
4. [Embeddings](https://github.com/GeorgeBatch/nlp_from_dls/tree/main/04_Embeddings)
    1. Lecture
        * Word2Vec: Skip-gram
        * Word2Vec: CBOW
        * GloVe
        * FastText
    2. Seminar
        * todo: add the description
5. [Coursework: Embeddings](https://github.com/GeorgeBatch/nlp_from_dls/tree/main/05_HW_Embeddings)
    * todo: add the description
6. [RNNs](https://github.com/GeorgeBatch/nlp_from_dls/tree/main/06_RNNs)
    1. Lecture
        * RNNs - some maths behind
        * Exploding and Vanishing gradient problems
        * LSTM and GRU overview
        * Examples of inputs and outputs
    2. Seminar
        * todo: add the description
7. [Coursework: Text Classification](https://github.com/GeorgeBatch/nlp_from_dls/tree/main/07_HW_Text_classification)
    * todo: add the description
8. [Language Models](https://github.com/GeorgeBatch/nlp_from_dls/tree/main/08_Language_models)
    1. Lecture
        * Count-based language models
            * Estimating probabilities in N-gram language model
            * Text-generation pipeline
            * Evaluating the quality of language models: Cross-entropy, Perplexity, human evaluation
        * Neural Language models
            * Using learnt or pre-computed embeddings
            * Using RNNs instead of Embeddings
            * Using CNNs + embeddings instead of RNNs
    2. Seminar
        * todo: add the description
9. [Coursework: Part-of-speech Tagger](https://github.com/GeorgeBatch/nlp_from_dls/tree/main/09_HW_Part_of_speech_tagger)
    * todo: add the description
10. [Neural Machine Translation](https://github.com/GeorgeBatch/nlp_from_dls/tree/main/10_Neural_Machine_Translation)
    1. Lecture
        * History of Machine Translation
        * Conditional language model
        * Encoder-decoder paradigm for MT
        * Seq2Seq model
        * BLEU (BiLingual Evaluation Understudy)
        * Training: Teacher Forcing
        * Problems and remedies of encoder-decoder architecture
            * Greedy decoding -> Beam Search
            * Forgetfullness of RNNs (even bidirectional with LSTM cells) -> Attention
    2. Seminar
        * Attention between encoder and decoder
        * Intro to self-attention and multi-head attention
11. [Coursework: Seq2Seq with Attention](https://github.com/GeorgeBatch/nlp_from_dls/tree/main/11_HW_Seq2Seq_with_Attention)
    * todo: add the description
12. [Transformers: Attention is all you need.](https://github.com/GeorgeBatch/nlp_from_dls/tree/main/12_Transformers)
    1. Lecture
        * Self-attention
        * Multi-head attention
        * Masked multi-head attention (general)
        * Positional encoding
        * Decoder Side: Masked multi-head attention from decoder to encoder
        * Learning rate: warm-up and cool-down stages

    2. Seminar
        * todo: add the description
13. [Transformers+](https://github.com/GeorgeBatch/nlp_from_dls/tree/main/13_Transformers%2B)
    1. Lecture
        * todo: add the description
    2. Seminar
        * todo: add the description
14. Coursework: Fine-tuning GPT and BERT
    * todo: add the description
----

**Disclaimer.** The course, in my opinion, is very similar to the [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) course from [Stanford](https://www.stanford.edu). Hence, the title and the description.
