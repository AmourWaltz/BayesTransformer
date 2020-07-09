# Transformer

​		本周主要学习了基于attention机制的Transformer语言模型，阅读相关文献并使用Python实现其基本框架，包括制作PTB语料库训练数据集，以及租用网络服务器完成训练。此外，在CS224n的NLP课程中还学习了Skip-gram以及CBOW两种word2vec模型并进行了简单实现。

## Schedule

* 7.2 - 7.8

## Questions

* 那天晚上问学长的关于Transformer是否传入全文本的max sentence length，后来想了想应该是positional embedding是传入max sentence length。但是如果使用时遇到比训练集里更长的句子，那这种positional embedding性能是否下降？
* 学习了两种word2vec模型，好奇既然这两种模型可以表达词向量之间的相似度，为什么不用于辅助语言模型预测？我理解的，像Transformer的word embedding包含较多序列信息，而word2vec的word embedding更倾向词义信息，有没有将两种word embedding结合使用的？
* 学长data.py里的一个小bug(其实也不算bug因为不影响结果，只是我输出后发现有点不对，强迫症所致，学长可以忽略)，# line 102 学长应该是想 if voc.word2id(token) + 1 < len(voc.idx2word)，要不然无法进入else，虽然index sequence还是正确的所以对训练没影响，但是文本中应该改为<unk>的地方最后没改。
* 论文：Character-Level Language Modeling with Deeper Self-Attention里的三种辅助损失函数，没太想明白原理或推导过程是什么，是否可以理解为是添加的损失函数正则项。

* Python的Argparse模块以前没用过，也没通过该模块执行命令行代码，一开始不会运行学长代码，现在也在学习这一块。
* 昨晚检查模型，decoder的sequence mask应该没问题，暂时还没找到问题出在哪，我自己电脑迭代一次都很慢，租用网络服务器有点麻烦。今天下午再检查一遍模型然后租服务器调试。