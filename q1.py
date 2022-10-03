from sklearn.feature_extraction.text import TfidfVectorizer
import math
corpus = [
    "Water - South Carolina",
    "Evolving Firearm Regulations",
    "Crime analysis in South Carolina",
    "Target aspect based sentiment analysis for urban neighborhoods",
    "Extracting synthesis procedure from solar cell perovskite based scientific publications.",
    "Entity Recognition : Water Data Regulations",
    "TOS: Banks' Terms of Services summary",
    "Water Regulation Summarization",
    "Predicting the 2022 gubernatorial election of South Carolina using sentiment analysis of Twitter.",
    "Scientific Artical Summarization",
    "New FastText [with Election data]",
    "Chatbot to answer quesries regarding WHO Water Regulations",
    "Verifying various foods connection to improve diabetes using NLP techniques",
    "Summarization of Terms and conditions",
    "Chatbot for Elections FAQ - State of Mississippi",
    "Image Captioning using Transformer Models",
    "Specialist Doctor Recommendation System",
    "Application of Artificial Neural Networks (ANN) to Automatic Speech Recognition (ASR) on a Novel Dataset created using YouTube",
    "Detecting and rating severity of urgency in short, one-time crisis events vs. ongoing ones",
    "Water Regulations - Arizona",
    "Damaged doc. prediction (10%)",
    "Visual Question Answering",
]

words= []
# 对corpus分词
for i in corpus:
    words.append(i.split())

    # 如果有自定义的停用词典，我们可以用下列方法来分词并去掉停用词
    # f = ["is", "the"]
    # for i in corpus:
    #     all_words = i.split()
    #     new_words = []
    #     for j in all_words:
    #         if j not in f:
    #             new_words.append(j)
    #     words.append(new_words)
    # print(words)

    # 进行词频统计


def Counter(word_list):
    wordcount = []
    for i in word_list:
        count = {}
        for j in i:
            if not count.get(j):
                count.update({j: 1})
            elif count.get(j):
                count[j] += 1
        wordcount.append(count)
    return wordcount


wordcount = Counter(words)


# 计算TF(word代表被计算的单词，word_list是被计算单词所在文档分词后的字典)
def tf(word, word_list):
    return word_list.get(word) / sum(word_list.values())


# 统计含有该单词的句子数
def count_sentence(word, wordcount):
    return sum(1 for i in wordcount if i.get(word))


# 计算IDF
def idf(word, wordcount):
    return math.log(len(wordcount) / (count_sentence(word, wordcount) + 1))


# 计算TF-IDF
def tfidf(word, word_list, wordcount):
    return tf(word, word_list) * idf(word, wordcount)


p = 1
for i in wordcount:
    print("part:{}".format(p))
    p = p + 1
    for j, k in i.items():
        print("word: {} ---- TF-IDF:{}".format(j, tfidf(j, i, wordcount)))

# 调包模板
#  https://blog.csdn.net/liuxuejiang158blog/article/details/31360765
import jieba
import jieba.posseg as pseg
import os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
if __name__ == "__main__":
    '''
   
    '''
    corpus = [
        "Water - South Carolina",
        "Evolving Firearm Regulations",
        "Crime analysis in South Carolina",
        "Target aspect based sentiment analysis for urban neighborhoods",
        "Extracting synthesis procedure from solar cell perovskite based scientific publications.",
        "Entity Recognition : Water Data Regulations",
        "TOS: Banks' Terms of Services summary",
        "Water Regulation Summarization",
        "Predicting the 2022 gubernatorial election of South Carolina using sentiment analysis of Twitter.",
        "Scientific Artical Summarization",
        "New FastText [with Election data]",
        "Chatbot to answer quesries regarding WHO Water Regulations",
        "Verifying various foods connection to improve diabetes using NLP techniques",
        "Summarization of Terms and conditions",
        "Chatbot for Elections FAQ - State of Mississippi",
        "Image Captioning using Transformer Models",
        "Specialist Doctor Recommendation System",
        "Application of Artificial Neural Networks (ANN) to Automatic Speech Recognition (ASR) on a Novel Dataset created using YouTube",
        "Detecting and rating severity of urgency in short, one-time crisis events vs. ongoing ones",
        "Water Regulations - Arizona",
        "Damaged doc. prediction (10%)",
        "Visual Question Answering",
    ]

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(corpus))
    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    for i in range(len(weight)):
        print("The", i, "weights------")
        for j in range(len(word)):
           print(word[j], weight[i][j])



    #Initialize an instance of tf-idf Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Generate the tf-idf vectors for the corpus
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # compute and print the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    for i in range(len(cosine_sim)):
        for j in range(len(cosine_sim[0])):
            if cosine_sim[i][j] > 0.9 and i != j:
                print(cosine_sim[i][j])






    k = KMeans(n_clusters=5)
    kmeans = k.fit(weight)
    print(kmeans.labels_)