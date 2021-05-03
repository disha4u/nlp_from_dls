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
        * tokenization, stop words, stemming, lemmatization, TF-IDF, cosine similarity
        * Text classification using all the above
        * NLP Libraries: nltk, [razdel](https://github.com/natasha/razdel), pymorphy2, spacy, rnnmorph 
3. [Coursework: Simple Embeddings](https://github.com/GeorgeBatch/nlp_from_dls/tree/main/03_HW_Simple_embeddings)
    * Ranking questions by similarity using
        * Self-implemented Tokenizer class and sentence-embedding aggregator function
        * Self-implemented Hits@K and DCG@K metrics
        * Word2Vec from gensim.models
    * NLP Libraries: gensim
4. [Embeddings](https://github.com/GeorgeBatch/nlp_from_dls/tree/main/04_Embeddings)
    1. Lecture
        * Word2Vec: Skip-gram
        * Word2Vec: CBOW
        * GloVe
        * FastText
    2. Seminar
        * Going deeper into Word2Vec implementation
        * Exploring embedding spaces:
            * Woman + King - Man = Queen
            * or less known: Mexico + Vodka - Russia = Tekila
        * Loonking into the biases introduced by the taining data
5. [Coursework: Embeddings](https://github.com/GeorgeBatch/nlp_from_dls/tree/main/05_HW_Embeddings)
    * Trying to come up and implement better embeddings for words not seen in the training set
        * Naive Context: average from the words to hte left and to the right
        * Existing Context: take the closest words to the left and right with known embeddings
        * Existing Context within a window of pre-specified length (no point in looking too far)
        * Same as before, but weighted inversely-proportionally to the distance
        * Previous concatenated with (TF-IDF + SVD) embedding
    * Training binary classification models with the above embeddings
6. [RNNs](https://github.com/GeorgeBatch/nlp_from_dls/tree/main/06_RNNs)
    1. Lecture
        * RNNs - some maths behind
        * Exploding and Vanishing gradient problems
        * LSTM and GRU overview
        * Examples of inputs and outputs
    2. Seminar
        * Going through implementations of different models for text classifications (for all models use learnable embedding layers)
            * CNN, RNN, GRU feature extraction models
            * Adding pre-trained word2vec embeddings and fine-tuning the models.
7. [Coursework: Text Classification](https://github.com/GeorgeBatch/nlp_from_dls/tree/main/07_HW_Text_classification)
    * Modifying models from seminar notebook to fit different dataset (multiclass classification)
    * Train the models optimizing towards self-implemented F1 score 
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
        * Going through the code for the models discussed un the lecture
        * Creating batches of similar size (`batch_size` * `seq_len`) from sentences of varying lengths
        * Temperature softmax, beam search
9. [Coursework: Part-of-speech Tagger](https://github.com/GeorgeBatch/nlp_from_dls/tree/main/09_HW_Part_of_speech_tagger)
    * Implementing and fitting HMM model for Part-of-speech (POS) tagging
    * Exploring default POS taggers from nlp libraries and their quality 
    * Implementing and fitting Bi-directional LSTM tagger
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
