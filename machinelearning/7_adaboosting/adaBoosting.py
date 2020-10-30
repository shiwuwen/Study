import numpy as np

#简单数据集上应用adaBoosting
def loadSimpData():
	'''
	创建简单数据集
	'''
	dataMat = np.matrix([[1., 2.1],
						[2., 1.1],
						[1.3, 1 ],
						[1. , 1.],
						[2. , 1.]])
	classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

	return dataMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
	'''
	使用单层决策树进行分类
	'''
	retArray = np.ones((np.shape(dataMatrix)[0], 1))

	if threshIneq == 'lt':
		#<=阈值 预测为-1了类
		retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
	else:
		#>阈值 预测为-1类
		retArray[dataMatrix[:, dimen] > threshVal] = -1.0

	return retArray


def buildStump(dataArr, classLabels, D):
	'''
	单层决策树算法
	'''
	dataMatrix = np.mat(dataArr)
	labelMat = np.mat(classLabels).T 
	m, n = np.shape(dataMatrix)

	#用于确定连续值步长的划分
	numSteps = 10.0
	bestStump = {}
	#预测结果
	bestClasEst = np.mat(np.zeros((m, 1)))
	minError = float('inf')

	#遍历属性，选择最优的属性进行划分
	for i in range(n):
		rangeMin = dataMatrix[:,i].min()
		rangeMax = dataMatrix[:,i].max()
		#将连续的属性值离散化
		stepSize = (rangeMax - rangeMin) / numSteps

		for j in range (-1, int(numSteps)+1):
			for inequal in ['lt', 'gt']:
				threshVal = (rangeMin + float(j)*stepSize)
				predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)

				errArr = np.mat(np.ones((m ,1)))
				#分类正确的置为0
				errArr[predictedVals == labelMat] = 0

				#基于权重的误差 1
				weightedError = D.T * errArr

				# print('split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f' %(i, threshVal, inequal, weightedError))

				if weightedError < minError:
					minError = weightedError
					bestClasEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal

	return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
	#保存训练的所有弱分类器
	weakClassArr = []
	m = np.shape(dataArr)[0]
	#初始化样本权重 m,1
	D = np.mat(np.ones((m,1)) / m)
	#记录每个数据的类别估计累计值
	aggClassEst = np.mat(np.zeros((m,1)))

	for i in range(numIt):
		# print('iter: ', i)

		bestStump, error, classEst = buildStump(dataArr, classLabels, D)
		# print('D: ', D.T)

		#计算弱分类器的权重
		alpha = float(0.5*np.log((1.0-error)/max(error, 1e-16)))

		bestStump['alpha'] = alpha
		weakClassArr.append(bestStump)
		# print('classEst: ', classEst.T)

		expon = np.multiply(-1*alpha*np.mat(classLabels).T, classEst)
		D = np.multiply(D, np.exp(expon))
		D = D / D.sum()

		#带权重的预测结果
		aggClassEst += alpha*classEst
		# print('aggClassEst: ', aggClassEst.T)

		aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m,1)))
		errorRate = aggErrors.sum() / m
		# print('total error: ', errorRate)

		if errorRate == 0:
			break

	return weakClassArr


def adaClassify(dataToClass, classifierArr):
	'''
	使用adaboosting进行预测
	'''
	dataMatrix = np.mat(dataToClass)
	m = np.shape(dataMatrix)[0]
	#基于权重的累计预测结果
	aggClassEst = np.mat(np.zeros((m,1)))

	for i in range(len(classifierArr)):
		classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
		aggClassEst += classifierArr[i]['alpha'] * classEst

		# print('prediect result: ', aggClassEst)

	#np.sign(input) 返回与input大小相同的output，且正数=1，0=0，负数=-1
	return np.sign(aggClassEst)



#在复杂数据集上应用adaBoosting
def loadDataSet(filename):
	'''
	加载filename
	'''
	#获取filename中的总行数
	numFeat = len(open(filename).readline().split('\t'))
	dataMat = []
	labelMat = []

	fr = open(filename)
	for line in fr.readlines():
		lineArr = []
		curLine = line.strip().split('\t')
		for i in range(numFeat-1):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))

	return dataMat, labelMat


def testAdaBoostingOnHardDataset(trainFilename, testFilename, numIt=40):
	'''
	在负载数据集上测试adaboosting
	'''
	dataArr, labelArr = loadDataSet(trainFilename)
	
	#获取n个弱分类器
	classifierArray = adaBoostTrainDS(dataArr, labelArr, numIt)

	testArr, testLabelArr = loadDataSet(testFilename)

	#获取测试结果
	predictionnumIt = adaClassify(testArr, classifierArray)

	errArr = np.mat(np.ones((len(testArr), 1)))
	errorRateNum = errArr[predictionnumIt != np.mat(testLabelArr).T].sum()

	errorRate = errorRateNum / len(testArr)

	#保留两位小数并返回
	return	round(errorRate, 2)












if __name__ == '__main__':

	# dataMat, classLabels = loadSimpData()

	# D = np.mat(np.ones((5, 1))/5)

	# bestStump, minError, bestClasEst = buildStump(dataMat, classLabels, D)

	# print(bestStump, minError, bestClasEst)

	# classifierArray = adaBoostTrainDS(dataMat, classLabels, 9)
	# print(classifierArray)

	# classifyRes = adaClassify([0,0], classifierArray)
	# print(classifyRes)


	#复杂数据集
	errorRate = testAdaBoostingOnHardDataset('horseColicTraining2.txt', 'horseColicTest2.txt', 50)
	print(errorRate)