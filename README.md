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
        * Implemented the model from the (2014) [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) paper commonly known as Seq2Seq
        * Intro to attention between encoder and decoder without implementation (see [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473))
        * Intro to self-attention and multi-head attention without implementation (see [Attention Is All You Need](https://arxiv.org/abs/1706.03762))
11. [Coursework: Seq2Seq with Attention](https://github.com/GeorgeBatch/nlp_from_dls/tree/main/11_HW_Seq2Seq_with_Attention)
    * Implemented something inbetween [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) and [GNMT-2016](https://arxiv.org/abs/1609.08144)
    * Achieved BLEU of 31 on Russian -> English translation
12. [Transformers: Attention is all you need.](https://github.com/GeorgeBatch/nlp_from_dls/tree/main/12_Transformers)
    1. Lecture
        * Self-attention, Multi-head attention, Masked multi-head attention (general)
        * Positional encoding
        * Decoder Side: Masked multi-head attention from decoder to encoder
        * Learning rate: warm-up and cool-down stages
    2. Seminar
        * Re-implement transformer architecture from [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
        * Train the model on the dataset from previous HW and achieve similar BLEU in 6x less time
13. [Transformers+](https://github.com/GeorgeBatch/nlp_from_dls/tree/main/13_Transformers%2B)
    1. Lecture
        * Contextual embeddings of [ELMO](https://arxiv.org/abs/1802.05365) vs Word2Vec
        * Cover what makes [BERT](https://arxiv.org/abs/1810.04805) and [GPT](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf) different from the original transformer and what they can do
    2. Seminar
        * Re-implementing GPT architecture from scratch
        * Looking at attention maps of pre-trained GPT from Hugging Face
        * Discuss the BPE (Byte Pair Encoding) algorithm. See [paper](https://paperswithcode.com/method/bpe), [blog post](https://leimao.github.io/blog/Byte-Pair-Encoding/)
        * Try using pre-trained GPT-2 for text generation, pre-trained DistilBERT for classification like shown in Jay Alamar's blog posts ([GPT-2](http://jalammar.github.io/illustrated-gpt2/), [DistilBERT](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/))
14. [Coursework: Fine-tuning GPT and BERT](https://github.com/GeorgeBatch/nlp_from_dls/tree/main/14_HW_Fine-Tuning_GPT_and_BERT)
    * Training GPT for text classification (accuracy=86%)
    * Fine-tuning pre-trained GPT for text classification (accuracy=92%)
    * Using DistilBERT for text classification on [SST](https://paperswithcode.com/dataset/sst): fine-tuned on SST (accuracy=86%), pre-trained on SST (accuracy=86%).
----

**Disclaimer.** The course, in my opinion, is very similar to the [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) course from [Stanford](https://www.stanford.edu). Hence, the title and the description.
