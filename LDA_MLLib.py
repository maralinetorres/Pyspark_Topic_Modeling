#MET CS777-  Big Data Analytics Final Project - LDA using library


from __future__ import print_function
import sys
from pyspark import SparkContext
import pandas as pd
import numpy as np
import time
import pickle
import re
#import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.clustering import LDA
import itertools

# nltk.download('wordnet')
# nltk.download('stopwords')


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: Topic Modeling <Dataset> <Topics_Output> <Topic Score_output>", file=sys.stderr)
        exit(-1)
    
    start_time = time.time()
    sc = SparkContext()

    # 1) Read the data from pickle file 
    #This data was prepared using python and you can find the code in the jupyter notebook
    print('Read the data from pickle file')
    m_read_file = open(sys.argv[1],"rb")
    motions_op = pickle.load(m_read_file)
    rdd = sc.parallelize(motions_op.values()).zipWithIndex().map(lambda x: (x[1], x[0]))

    #2) Tokenization, removing stop words and lemmatize
    print('Tokenization, removing stop words and lemmatize')
    regex = re.compile('[^a-zA-Z]')
    d_keyAndListOfWords = rdd.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

    STOPWORDS = list(stopwords.words('english'))
    #added some words that are repetitive in the motions
    STOPWORDS.extend(['whereas','regard','member','parliament','council','calls','commission','rights','right','europe','european'])

    def removeStopWordsShortWordsLem(ListOfWords):
        result = []
        wnl = WordNetLemmatizer()
        for word in ListOfWords:
            if (word not in STOPWORDS and len(word) > 2):
                result.append(wnl.lemmatize(word))
        return result

    d_keyAndListOfWords_clean = d_keyAndListOfWords.map(lambda x: (x[0], removeStopWordsShortWordsLem(x[1])))

    #3) Preparing 5k dictionary
    print('Preparing 5k Dictionary')
    allWordsCounts = d_keyAndListOfWords_clean.flatMap(lambda x: x[1]).map(lambda x: (x,1)).reduceByKey(lambda x,y: x+y)
    # Getting the top 5,000 words in a local array in a sorted format based on frequency
    topWords = allWordsCounts.top(5000, lambda x: x[1])
    topWordsK = sc.parallelize(range(5000))
    dictionary = topWordsK.map(lambda x:(topWords[x][0], x))

    #4) Tokenization
    print('Creating tokenizations...')
    def buildCountWordsArray(listOfIndexes):
        counts = pd.Series(listOfIndexes).value_counts()
        x5K = pd.Series(np.zeros(5000).tolist())
        y = pd.concat([x5K, counts], axis=1, join='outer')
        y.columns = ['Mock','Real']
        y.drop(columns=['Mock'], inplace=True)
        y.fillna(0, inplace=True)
        return y.Real.values

    # Next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
    # ("word1", docID), ("word2", docId), ...
    allWordsWithDocID = d_keyAndListOfWords_clean.flatMap(lambda x: ((j, x[0]) for j in x[1]))

    # Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
    dictionary_small_dataset = dictionary.collectAsMap()
    sc.broadcast(dictionary_small_dataset)
    #Let's do a simple map on it
    allDictionaryWords=allWordsWithDocID.map(lambda x: (x[0], (dictionary_small_dataset.get(x[0]), x[1])) if x[0] in dictionary_small_dataset.keys() else None).filter(lambda x: x!=None)

    # Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
    justDocAndPos = allDictionaryWords.map(lambda x: (x[1][1],x[1][0]))
    # Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
    allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()
    # (Document Id and [Vector of counts])
    resultLDA = allDictionaryWordsInEachDoc.mapValues(list).map(lambda x: [int(x[0]), Vectors.dense(buildCountWordsArray(x[1]))])  
   


    #5) Starting with the LDA
    data_lda = resultLDA.filter(lambda x: x[0] in np.arange(4500).tolist())
    start_LDAtime = time.time()
    lda_model = LDA.train(data_lda, k=10, maxIterations=100)
    print("Total Time to run the LDA Code: --- %s minutes ---" % round((time.time() - start_LDAtime)/60,3))  


    #6) Extracting the process
    inv_vocabulary = {v: k for k, v in dictionary_small_dataset.items()}
    lda_topics = lda_model.topicsMatrix()
    number_top_words = 10
    lda_topicWords = {}
    for num in range(10):
        top20Words = sc.parallelize(lda_topics[:,num]).zipWithIndex().top(number_top_words, lambda x: x[0])
        wordsInTopic = []
        for quantIndex in top20Words:
            word = quantIndex[1]
            wordsInTopic.append(inv_vocabulary[word])
            lda_topicWords[num] = wordsInTopic
        print(f'Topic #{num} : {" ".join(wordsInTopic)}')

    #7) Evaluate the topic coherence
    def getDictionaryIndex(listOfWords):
        listIndexes = []
        for word in listOfWords:
            listIndexes.append(dictionary_small_dataset[word])
        return listIndexes

    def correctDocumentsLDA(p, w0, w1=0):
        exists = False
        if w1 == 0:
            for w in p:
                if w == w0:
                    exists = True
        else:
            exists = all(x in p for x in [w0,w1])
        return exists

    def convertToIndexes(p):
        words = []
        #get words that really exist in the document
        for n, v in enumerate(p[1]):
            if v != 0:
                words.append(n)
        return (p[0], words)
        
    def getDwiLDA(topic10Words):
        topicIndexes = getDictionaryIndex(topic10Words)
        document_wordCount = {}
        count = 0
        for index in topicIndexes:
            #count = rdd_converted.filter(lambda x: correctDocuments(x, index)).count()
            for key, value in dict_LDA.items():
                count += correctDocumentsLDA(value, index)
            document_wordCount[index] = count
        return document_wordCount

    def getDwiWjLDA(topic10Words, word):
        allCombinations = list(itertools.combinations(topic10Words,2))
        combinationCount = {}
        count = 0
        for cmb in allCombinations:
            if word == cmb[0]:
                wordsComb = getDictionaryIndex(list(cmb))
                #count = rdd_converted.filter(lambda x: correctDocuments(x, wordsComb[0], wordsComb[1])).count()
                for key, value in dict_LDA.items():
                    count += correctDocumentsLDA(value, wordsComb[0], wordsComb[1])
                dictIndex = str(wordsComb[0]) + "," + str(wordsComb[1])
                combinationCount[dictIndex] = count
        return combinationCount

    def calculateTopicCoherenceLDA(listOfWords):
        allDwi = getDwiLDA(listOfWords)
        score = {}
        for key, val in allDwi.items():
            dwiWj = getDwiWjLDA(listOfWords, inv_vocabulary[key]) #get the dwiWj
            for pairs, count in dwiWj.items():
                cal = np.log((count + 1)/val) #do the calculation from the above formula
                score[pairs] = cal
        return np.sum(list(score.values()))

    rdd_converted = data_lda.map(convertToIndexes)
    topicScoreLDA = {}
    dict_LDA = rdd_converted.collectAsMap()
    for num, words in lda_topicWords.items():
        #print(num, words)
        coherence = calculateTopicCoherenceLDA(words)
        indexing = f'Topic #{num}'
        topicScoreLDA[indexing] = coherence
    
    #8) Save the results
    print('Saving output file')
    rddTopics = sc.parallelize(lda_topicWords.items()).coalesce(1)
    rddTopics.saveAsTextFile(sys.argv[2])

    rddTopicScore = sc.parallelize(topicScoreLDA.items()).coalesce(1)
    rddTopicScore.saveAsTextFile(sys.argv[3])


    print('Script done')

    sc.stop()

    print("Total Time to run the Script: --- %s minutes ---" % ((time.time() - start_time)/60))