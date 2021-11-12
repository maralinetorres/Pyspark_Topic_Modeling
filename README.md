# Introduction

This repository was created to present one of my Big Data Analytic project.


## Project Description

The goal of this project was to implement Latent Dirichlet Allocation (LDA) from scratch to do Topic Modeling. The dataset used was obtained from the Digital Corpus of the European Parliament (DCEP). The DCEP contains most of the documents published on the European Parliament's official website from press releases to session and legislative documents related to European Parliament's activities and bodies. However, for the project, I focused on the Documents that are considered “Motions” and that are written in English. The version downloaded and pre-processed contains documents that were produced between 2001 and 2012 which totals 66,607 motions. 

The DCEP provides the documents as XML files. The jupyter notebook in docs folders, contains the code used to clean and combine all the XML files into a pickle file. 

**Objective:** What are the 10 topics discussed in DCEP motions?

**Dataset:** Digital Corpus of the European Parliament (DCEP) dataset

**Topic Modeling technique:** Latent Dirichlet Allocation (LDA) using Gibbs Sampling

**Number of documents processed:** 3,000 

**Evaluation metric:** Topic Coherence (Intrinsic UMass Measure)

# Python scripts

1) **LDA_scratch.py** - The goal of this script was to implement LDA from scratch using Gibbs Sampling. It prepares the data by doing tokenization, lemmazitation and removing stops words. Also, it creates the 5k Dictionary and  the final RDD with the bag of words. Then, it runs 50 iterations to 3,000 motions to identify 10 topics. The hyperparameters alpha and beta we both initialized with 0.1. Then, it extracts the topics by taking the top 10 words assigned to each one of them and calculates the topic coherence for each topic. It outputs the topics and topic coherence score. 

2) **LDA_MLLib.py** - The goal of this script was to implement LDA using Pyspark Library and evaluate the model performance. Similar as the first script, it prepares the data and creates the Document Term Frequency to train the model.  Trained the LDA with 4,500 documents to identify 10 topics. It outputs the topics identified and it's respective topic coherence. 


# Other Documents. 

The doc folder contains:
  - Output files folder
  - Final Project Jupyter notebook 
  - Dataset - EuroParl-motions.pkl
  

# How to run  

To run this program, you will need the following files:

1. The Python Script (Example: **LDA_scratch.py**)
2. The input(s) data file(s) (Example: **EuroParl-motions**)
3. The output files (Example: **~/10Topics**)

```python

spark-submit LDA_scratch.py <EuroParl-motions_inputfile> <Topics_output> <TopicsScore_outputfile>

```



```python

spark-submit LDA_MLLib.py <EuroParl-motions_inputfile> <Topics_outputfile> <TopicsCoherence_outputfile>

```