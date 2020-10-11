#coding=utf-8

import numpy as np
import math

def loadDataSet():
	postingList = [	['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
					['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
					['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
					['stop', 'posting', 'stupid', 'worthless', 'garbage'],
					['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
					['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0,1,0,1,0,1]
	
	return postingList, classVec


def createVocabList(dataset):
	vocabSet = set([])
	for document in dataset:
		vocabSet = vocabSet | set(document)

	return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print('the word: %s is not in the vocabulary' % word)

	return returnVec	


def trainNB0(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory) / float(numTrainDocs)
	
	#防止概率0
	# p0Num = np.zeros(numWords)
	# p1Num = np.zeros(numWords)
	# p0Denom = 0
	# p1Denom = 0
	p0Num = np.ones(numWords)
	p1Num = np.ones(numWords)
	p0Denom = 2.0 #2指代当前属性可能的取值有两种
	p1Denom = 2.0


	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else: 
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])

	#防止下溢
	p1Vect = np.log(p1Num / p1Denom)
	p0Vect = np.log(p0Num / p0Denom)

	return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	p1 = sum(vec2Classify*p1Vec) + math.log(pClass1)
	p0 = sum(vec2Classify*p0Vec) + math.log(1.0 - pClass1)

	if p1>p0:
		return 1
	else:
		return 0

def testingNB():
	listOPosts, listClasses = loadDataSet()
	myVocabList = createVocabList(listOPosts)
	trainMat = []

	for postinDoc in listOPosts:
		trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

	#传入函数的是np.ndarray
	p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))

	testEntry = ['love', 'my', 'dalmation']
	thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
	print(testEntry, 'classified as ', classifyNB(thisDoc, p0V, p1V, pAb))

	testEntry = ['stupid', 'garbage']
	thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
	print(testEntry, 'classified as ', classifyNB(thisDoc, p0V, p1V, pAb))



if __name__ == '__main__':
	testingNB()