#MET CS777-  Big Data Analytics Final Project - LDA from scratch


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
import itertools

# nltk.download('wordnet')
# nltk.download('stopwords')


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: Topic Modeling <Dataset> <Topic output> <TopicScore_output>", file=sys.stderr)
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

    #4) Creating the bag of words
    print('Creating bag of words')
    def buildBOWTuple(docId, listOfIndexes):
        values = pd.Series(listOfIndexes).value_counts().values
        numWords = np.sum(values)
        return (docId, listOfIndexes,numWords)

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
    # (Document Id, [List of indexes], number of words in that document)
    result = allDictionaryWordsInEachDoc.mapValues(list).map(lambda x: buildBOWTuple(x[0], x[1]))

    motions = result.map(lambda x: (x[1],x[2])).zipWithIndex().map(lambda x: (x[1], x[0][0], x[0][1]))

    #5) Starting with the LDA
    print('Starting with LDA')
    result = motions.filter(lambda x: x[0] in np.arange(3000).tolist()).map(lambda x: (x[1],x[2])).zipWithIndex().map(lambda x: (x[1], x[0][0], x[0][1]))
    result.cache()
    topics = 10

    alpha = 0.1        # the parameter of the Dirichlet prior on the per-document topic distributions
    beta = 0.1      # the parameter of the Dirichlet prior on the per-topic word distribution

    NumDocs = result.count() #Number of documents
    vocab = dictionary.map(lambda x: x[0]).collect()

    cTW = np.zeros((topics, len(vocab)))  #the number of times the word is assigned to the topic
    cDT = np.zeros((NumDocs, topics)) #the count of of topic j assigned to some word token in document d

    n_d = np.zeros((NumDocs)) #the number of words of each document, we already have that in the rdd
    n_z = np.zeros((topics)) #the number of times any word is assigned to topic k

    ##Necessary functions

    def generateTopic(NumWords):
        n_w_t = [0 for _ in range(NumWords)]    
        for n in range(NumWords):
            n_w_t[n] = np.random.randint(0, topics)
        return n_w_t   

    def getTopicAssignedToWord(docId,numWords):
        topicsAssigned = []
        for n in range(numWords):
            z = z_i_j[docId][n]
            topicsAssigned.append(z)
        return topicsAssigned

    def buildcDT(topicsVals):
        n_d_t = np.zeros(topics)
        x, counts = np.unique(topicsVals, return_counts=True)
        n_d_t[x] = counts
        return n_d_t

    def buildcTopicWord(words, topics):
        wt = []
        for n, word in enumerate(words):
            wt.append((word, topics[n]))
        return wt

    #randomly assign topic to each word of each document
    z_i_j = result.map(lambda x: generateTopic(x[2])).collect()

    #(Document Id, {VocabIndex: Word occurence in document}, Number of words in document, Topic assigned to each word)
    rdd = result.map(lambda x: (x[0],x[1], x[2], getTopicAssignedToWord(x[0], x[2])))
    #Creating the initial document-topic distribution
    rdd1 = rdd.map(lambda x: (x[0], x[1], x[2],x[3],buildcDT(x[3]))) 
    cDT = np.stack(rdd1.map(lambda x: x[4]).collect())

    #Creatting the initial word-topic distribution
    #Pass List of words, topics
    ctw = rdd1.map(lambda x: buildcTopicWord(x[1], x[3])).flatMap(lambda x: x).map(lambda x: (x,1)).reduceByKey(lambda x,y: x+y).collect()
    for tw in ctw:
        cTW[tw[0][1], tw[0][0]] = tw[1]

    #Get the number of times any word is assigned to topic k (Nz)
    n_z = np.sum(cTW, axis=1) 

    def collapsedGibbsSampling(docId, doc, numWords, z_w_t, phi_wt, theta_d_t, n_z_topic):
        vocabulary = len(phi_wt[0])
        numTopics = len(phi_wt)
        alpha = 1 / numTopics
        beta = 1 / numTopics
        
        for n, word in enumerate(doc):
            #get topic
            z = z_w_t[n]
            
            # decrement counts for word w with associated topic z because now we are 
            #going to assign it a new topic
            theta_d_t[z]-= 1
            phi_wt[z, word]-=1
            n_z_topic[z]-=1

            #Calculate the new topic for the word using the formula presented above. 
            prob_dt = (theta_d_t + alpha) / (numWords - 1 + numTopics * alpha)
            prob_topic_w = (phi_wt[:, word] + beta) / (n_z_topic + vocabulary * beta)
            p_z = prob_dt * prob_topic_w
            p_z /= np.sum(p_z)
            new_z = np.random.multinomial(1, p_z).argmax()

            # Update the topic assignment for the word 
            #And increment count for the new topic assigned
            z_w_t[n] = new_z
            theta_d_t[new_z] += 1
            phi_wt[new_z, word] += 1
            n_z_topic[new_z] += 1

        return (docId, doc, numWords, z_w_t, theta_d_t, phi_wt, n_z_topic)

    print('Running the 50 iterations....')
    start_LDAtime = time.time()
    for iteration in (range(50)):
        print(f'Iteration {iteration}')   
        
        resultGibbs = rdd1.map(lambda x: collapsedGibbsSampling(x[0], x[1], x[2], x[3], cTW, x[4], n_z)).cache()
        
        rdd1 = resultGibbs.map(lambda x: (x[0],x[1], x[2], x[3], x[4])).cache()

        cTW = np.zeros((topics, len(vocab))) 
        #Pass List of words and topics assigned
        ctw = rdd1.map(lambda x: buildcTopicWord(x[1], x[3])).flatMap(lambda x: x).map(lambda x: (x,1)).reduceByKey(lambda x,y: x+y).collect()
        #Update the phi to get the new topic assigned to each word
        for tw in ctw:
            cTW[tw[0][1], tw[0][0]] = tw[1]
        
        #Update the nz (nd) to get the new number of times any word is assigned to topic k
        n_z = np.sum(cTW, axis=1) 

    print("Total Time to run the LDA Code: --- %s minutes ---" % round((time.time() - start_LDAtime)/60,3))

    finalcDT = np.stack(rdd1.map(lambda x: x[4]).collect())

    #6) Extract the topics
    print('Extracting the topics....')
    inv_vocabulary = {v: k for k, v in dictionary_small_dataset.items()}
    number_top_words = 10
    topicWords = {}
    for num, topic in enumerate(cTW):
        top20Words = sc.parallelize(topic).zipWithIndex().top(number_top_words, lambda x: x[0])
        wordsInTopic = []
        for quantIndex in top20Words:
            word = quantIndex[1]
            wordsInTopic.append(inv_vocabulary[word])
            topicWords[num] = wordsInTopic
        print(f'Topic #{num} : {" ".join(wordsInTopic)}')

    rddTopics = sc.parallelize(topicWords.items()).coalesce(1)

    #7) Evaluate the topics
    def getDictionaryIndex(listOfWords):
            listIndexes = []
            for word in listOfWords:
                listIndexes.append(dictionary_small_dataset[word])
            return listIndexes

    def correctDocuments(p, w0, w1=0):
        if w1 == 0:
            for w in p[1]:
                if w == w0:
                    return p
        else:
            exists = all(x in p[1] for x in [w0,w1])
            if exists:
                return p
            
    def getDwi(topic10Words):
        topicIndexes = getDictionaryIndex(topic10Words)
        document_wordCount = {}
        for index in topicIndexes:
            count = result.filter(lambda x: correctDocuments(x, index)).count()
            document_wordCount[index] = count
        return document_wordCount

    def getDwiWj(topic10Words, word):
        allCombinations = list(itertools.combinations(topic10Words,2))
        combinationCount = {}
        for cmb in allCombinations:
            if word == cmb[0]:
                wordsComb = getDictionaryIndex(list(cmb))
                count = result.filter(lambda x: correctDocuments(x, wordsComb[0], wordsComb[1])).count()
                dictIndex = str(wordsComb[0]) + "," + str(wordsComb[1])
                combinationCount[dictIndex] = count
        return combinationCount

    def calculateTopicCoherence(listOfWords):
        allDwi = getDwi(listOfWords)
        score = {}
        for key, val in allDwi.items():
            dwiWj = getDwiWj(listOfWords, inv_vocabulary[key]) #get the dwiWj
            for pairs, count in dwiWj.items():
                cal = np.log((count + 1)/val) #do the calculation from the above formula
                score[pairs] = cal
        return np.sum(list(score.values()))

    topicScore = {}
    for num, words in topicWords.items():
        #print(num, words)
        coherence = calculateTopicCoherence(words)
        indexing = f'Topic #{num}'
        topicScore[indexing] = coherence

    rddTopicScore = sc.parallelize(topicScore.items()).coalesce(1)

    ##9) Save output files
    print('Saving output file')
    rddTopics.saveAsTextFile(sys.argv[2])
    rddTopicScore.saveAsTextFile(sys.argv[3])

    print('Script done')

    sc.stop()

    print("Total Time to run the Script: --- %s minutes ---" % ((time.time() - start_time)/60))