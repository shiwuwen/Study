#coding=utf-8

import numpy as np
import math


#过滤网站恶意留言
def loadDataSet():
	#创造数据集
	postingList = [	['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
					['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
					['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
					['stop', 'posting', 'stupid', 'worthless', 'garbage'],
					['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
					['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	#类别标签，1-包含敏感词 0-不包含
	classVec = [0,1,0,1,0,1]
	
	return postingList, classVec


def createVocabList(dataset):
	'''
	确定dataset中包含的全部单词数
	'''
	#使用set集合去除重复单词
	vocabSet = set([])
	for document in dataset:
		# | 表示并集
		vocabSet = vocabSet | set(document)

	return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
	'''
	词集模型
	将单词装换成词向量，长度len(voacbList)
	'''
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print('the word: %s is not in the vocabulary' % word)

	return returnVec	


def bagOfWords2VecMN(vocabList, inputSet):
	'''
	词袋模型
	考虑单词出现的次数
	'''
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
		else:
			print('the word: {} is not in the vocabulary'.format(word))


def trainNB0(trainMatrix, trainCategory):
	'''
	使用Bayes算法对文本进行分类
	p(c|x) = p(x|c)*p(c)/p(x)
	在分类时无需计算p(x)
	'''
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	#p(c=1) p(c=0)=1-p(c=1)
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
		#p(x|c)
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
	'''
	文本分类
	'''
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


#使用朴素贝叶斯对垃圾邮件进行分类
def textParse(bigString):
	import re
	listOfTokens = re.split(r'\w*', bigString)
	return [tok.lower for tok in listOfTokens if len(tok)>2]


def spamTest():
	docList = []; classList = []; fullText = []
	for i in range(1,26):
		wordList = textParse(open('email/spam/%d.txt' %i, 'r', encoding='gb18030').read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)

		#使用utf-8编码报错
		wordList = textParse(open('email/ham/%d.txt' %i, 'r', encoding='gb18030').read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)

	vocabList = createVocabList(docList)

	trainingSet = list(range(50)); testSet = []
	for i in range(10):
		#np.random.choice(trainingSet)
		randIndex = int(np.random.uniform(0, len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])

	trainMat = []; trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])

	p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))

	errorCount = 0

	for docIndex in testSet:
		wordVector = setOfWords2Vec(vocabList, docList[docIndex])
		if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
			errorCount += 1
			# print("分类错误的测试集：", docList[docIndex])

	print('the error rate is: ', float(errorCount)/ len(testSet))


#使用朴素贝叶斯从个人广告中获取倾向



if __name__ == '__main__':
	# testingNB()

	spamTest()