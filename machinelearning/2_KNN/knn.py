#coding=utf-8

import numpy as np 
import operator
import matplotlib.pyplot as plt 
import matplotlib
from os import listdir


def creatDataSet():
	group = np.array([[1.0,1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	
	#tile(input, (x,y)) 将input 重复x行，y列
	diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistance = sqDiffMat.sum(1)
	distance = sqDistance**0.5

	#argsort默认升序排列（从小到大）
	sortedDistIndicies = distance.argsort()
	classCount = {}

	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		# dict.get(value, default=None) 获取字典中的value，若不存在则返回default
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	# python3中用items()替换python2中的iteritems()
    # key = operator.itemgetter(1)根据字典的值进行排序
    # key = operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    # sorted可以对任意可迭代数据类型进行排序
	sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

	return sortedClassCount[0][0]


def file2matrix(filename):
	fr = open(filename)
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)
	returnMat = np.zeros((numberOfLines,3))
	classLabelVetor = []
	index = 0

	for line in arrayOLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index, :] = listFromLine[:3]
		classLabelVetor.append(int(listFromLine[-1]))
		index += 1

	fr.close()
	return returnMat, classLabelVetor


def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = np.zeros(np.shape(dataSet))
	m = dataSet.shape[0]

	normDataSet = dataSet - np.tile(minVals, (m, 1))
	normDataSet = normDataSet/np.tile(ranges, (m, 1))

	return normDataSet, ranges, minVals


def datingClassTest():
	hoRatio = 0.1 
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0

	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i, :], normMat[numTestVecs:, :], datingLabels[numTestVecs:], 3)
		print('the test result is {}, the real result is {}'.format(classifierResult, datingLabels[i]))
		if classifierResult != datingLabels[i]:
			errorCount += 1

	print('the total error rate is %f'%(errorCount/numTestVecs))
 

def classifiyPerson():
	resultList = ['do not like', 'little like', 'much like']

	playingGames = float(input('玩游戏的所耗时间百分比:'))
	flyMiles = float(input('每年的飞行里程数：'))
	iceCream = float(input('每周吃的冰淇淋数：'))

	inX = np.array([playingGames, flyMiles, iceCream])

	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')

	normMat, ranges, minVals = autoNorm(datingDataMat)
	normInX = (inX - minVals) / ranges

	classifierResult = classify0(normInX, normMat, datingLabels, 3)

	print('she may be {} this person'.format(resultList[classifierResult-1]))



def img2vector(filename):
	returnVector = np.zeros((1, 1024))
	fr = open(filename)

	for i in range(32):
		line = fr.readline()
		for j in range(32):
			returnVector[0, 32*i+j] = int(line[j])

	fr.close()
	return returnVector


def handwritingClassTest():
	hwLabels = []
	trainingFileList = listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = np.zeros((m, 1024))
	for i in range(m):
		classNumber, trainingMat[i] = getNameClass(trainingFileList[i], 'trainingDigits')
		hwLabels.append(classNumber)

	testFileList = listdir('testDigits')
	errorCount = 0

	for j in range(len(testFileList)):
		classNumberTest, vectorUnderTest = getNameClass(testFileList[j], 'testDigits')
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print('the test result is {}, the real result is {}'.format(classifierResult, classNumberTest))
		if classifierResult != classNumberTest:
			errorCount += 1

	print('the error rate is %f' %(errorCount/len(testFileList)))


def getNameClass(filename, dirname):
	filestr = filename.split('.')[0]
	classNumber = int(filestr.split('_')[0])
	vector = img2vector('%s/%s'%(dirname, filename))

	return classNumber, vector


if __name__ == '__main__':
	# group, labels = creatDataSet()
	# result = classify0([0,0], group, labels, 3)
	# print(result)

	# datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	# normMat = autoNorm(datingDataMat) 
	# print(normMat)
	# print(labels)

	# datingClassTest()

	# classifiyPerson()

	# print(img2vector('testDigits/0_13.txt')[0, 0:31])

	handwritingClassTest()