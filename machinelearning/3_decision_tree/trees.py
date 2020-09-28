from math import log
import operator


def creatDataSet():
	'''
	返回数据以及属性值
	'''
	dataSet = [[0, 0, 0, 0, 'no'],
				[0, 0, 0, 1, 'no'],
				[0, 1, 0, 1, 'yes'],
				[0, 1, 1, 0, 'yes'],
				[0, 0, 0, 0, 'no'],
				[1, 0, 0, 0, 'no'],
				[1, 0, 0, 1, 'no'],
				[1, 1, 1, 1, 'yes'],
				[1, 0, 1, 2, 'yes'],
				[1, 0, 1, 2, 'yes'],
				[2, 0, 1, 2, 'yes'],
				[2, 0, 1, 1, 'yes'],
				[2, 1, 0, 1, 'yes'],
				[2, 1, 0, 2, 'yes'],
				[2, 0, 0, 0, 'no']]
    # 分类属性
	featureLabels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    # 返回数据集和分类属性
	return dataSet, featureLabels


def calcShannonEnt(dataSet):
	'''
	计算香农熵
	'''
	#数据集样本总个数
	numEntries = len(dataSet)
	labelCounts = {}

	for featVec in dataSet:
		#获取每一个样本的类别标签，并计数
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
		# labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
	shannonEnt = 0
	#计算香农熵
	for key in labelCounts:
		prob = float(labelCounts[key]) / numEntries
		shannonEnt -= prob*log(prob, 2)

	return shannonEnt


def splitDataSet(dataSet, axis, value):
	'''
	按照某一个属性的某一个取值划分数据集
	返回axis列的值为value的那些样本
	'''
	retDataSet = []

	for featVec in dataSet:
		if featVec[axis]==value:
			#返回的样本不报含axis列
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)

	return retDataSet


def chooseBestFeatureToSplit(dataSet):
	'''
	ID3算法
	使用信息增益选择最好的划分属性
	'''
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain = 0
	bestFeature = -1

	#计算每一个属性的信息增益，选择最大的属性最为当前划分
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)
		newEntropy = 0

		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = float(len(subDataSet)) / float(len(dataSet))
			newEntropy += prob*calcShannonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy
		if infoGain>bestInfoGain:
			bestInfoGain = infoGain
			bestFeature = i
	#返回的是信息增益最大的属性的下标，即以第几列作为当前的划分标准
	return bestFeature


def majorityCnt(classList):
	'''
	当数据集只有一个属性但样本类别不完全相同时，使用投票法确定当前类别
	'''
	classCount = {}
	for vote in classList:
		classCount = classCount.get(vote, 0) + 1
	sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

	return sortedClassCount[0][0]


def creatTree(dataSet, labels):
	'''
	使用递归算法创建决策树
	'''
	#数据集中的所有类别标签
	classList = [example[-1] for example in dataSet]
	if classList.count(classList[0])==len(classList):
		return classList[0]
	if len(dataSet[0])==1:
		return majorityCnt(classList)

	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatureLabel = labels[bestFeat]
	myTree = {bestFeatureLabel:{}}

	# subLabels = labels[:]
	# del(subLabels[bestFeat])
	# del(labels[bestFeat])
	

	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)

	for value in uniqueVals:
		#深拷贝
		# subLabels = labels[:]
		subDataSet = splitDataSet(dataSet, bestFeat, value)
		myTree[bestFeatureLabel][value] = creatTree(subDataSet, labels)

	return myTree


def classify(inputTree, featureLables, testVec):
	'''
	使用预先构建的决策树对输入进行预测
	'''
	firstStr = list(inputTree.keys())[0]
	secondDict = inputTree[firstStr]
	featureIndex = featureLables.index(firstStr)

	for key in secondDict.keys():
		if testVec[featureIndex] == key:
			if type(secondDict[key]).__name__ == 'dict':
				classLabel = classify(secondDict[key], featureLables, testVec)
			else:
				classLabel = secondDict[key]

	return classLabel	


def storeTree(inputTree, filename):
	'''
	将文件以字节流的形式保存
	打开文件时使用b模式
	'''
	import pickle
	fw = open(filename, 'wb')
	pickle.dump(inputTree, fw)
	fw.close()

def grabTree(filename):
	'''
	读取pickle序列化的python结构
	'''
	import pickle
	fr = open(filename, 'rb')
	return pickle.load(fr)


import copy

if __name__ == '__main__':
	dataSet, labels = creatDataSet()
	# featureLabels = copy.deepcopy(labels)
	# print(dataSet)
	

	# print(calcShannonEnt(dataSet))

	# print(chooseBestFeatureToSplit(dataSet))

	mytree = creatTree(dataSet, labels)
	print(mytree)

	# storeTree(mytree, './test.txt')

	# print(grabTree('./test.txt'))

	classLabel = classify(mytree, labels, [0,0,0,1])

	print(classLabel)