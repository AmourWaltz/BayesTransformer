# Weekly Report

* Reporter: XUE Bo-yang
* Schudule: 7.2 - 7.8
* Tasks: Transformer Language Model; Word2vec Models.
* Github: https://github.com/AmourWaltz/Transformer_testing
* The related literature notes is attached.

## Transformer

* In this week, I mainly learned the Transformer language model based on attention mechanism. After learning relevant paper, *Attention is all your need,* I implement a basic Transformer language model using Python.
* Corpus Making: PTB dataset is used to obtain corpus, which includes spliting sentences into words, establishing index for each word and converting snetences into index sequences. Then they are divided into training set, validation set and testing set. 
* Transformer Model: *Attention is all your need*  is referred to model, including Encoder, Decoder, Self-Attention, Positional Embedding, Masks and so on.
* Training: Renting an online server to complete training process.

## Word2vec

* In addition, I also learned two word2vec models of Skip-gram and CBOW  in NLP lectures of CS224n and simply practiced.
* Corpus Making: PTB dataset is used again to obtain corpus, which includes spliting sentences into words, establishing index for each word and extracting center words and contexts.
* Skip-gram Model: Firstly, the word embedding is established, and each element contains center, context, negative in data batch. After training, cosine similarity of two words represents the relevance of their senses.

## Reference

* *Attention is all your need* 
* *Character-Level Language Modeling with Deeper Self-Attention*

