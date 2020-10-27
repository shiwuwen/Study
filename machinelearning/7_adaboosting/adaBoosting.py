import numpy as np

#简单数据集上应用adaBoosting
def loadSimpData():
	dataMat = np.matrix([[1., 2.1],
						[2., 1.1],
						[1.3, 1 ],
						[1. , 1.],
						[2. , 1.]])
	classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

	return dataMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
	retArray = np.ones((np.shape(dataMatrix)[0], 1))

	if threshIneq == 'lt':
		retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:, dimen] > threshVal] = -1.0

	return retArray


def buildStump(dataArr, classLabels, D):
	dataMatrix = np.mat(dataArr)
	labelMat = np.mat(classLabels).T 
	m, n = np.shape(dataMatrix)

	numSteps = 10.0
	bestStump = {}
	bestClasEst = np.mat(np.zeros((m, 1)))
	minError = float('inf')

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
	weakClassArr = []
	m = np.shape(dataArr)[0]
	D = np.mat(np.ones((m,1)) / m)
	#记录每个数据的类别估计累计值
	aggClassEst = np.mat(np.zeros((m,1)))

	for i in range(numIt):
		# print('iter: ', i)

		bestStump, error, classEst = buildStump(dataArr, classLabels, D)
		# print('D: ', D.T)

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
	dataMatrix = np.mat(dataToClass)
	m = np.shape(dataMatrix)[0]
	aggClassEst = np.mat(np.zeros((m,1)))

	for i in range(len(classifierArr)):
		classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
		aggClassEst += classifierArr[i]['alpha'] * classEst

		# print('prediect result: ', aggClassEst)

	#np.sign(input) 返回与input大小相同的output，且正数=1，0=0，负数=-1
	return np.sign(aggClassEst)



#在复杂数据集上应用adaBoosting
def loadDataSet(filename):
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
	dataArr, labelArr = loadDataSet(trainFilename)
	classifierArray = adaBoostTrainDS(dataArr, labelArr, numIt)

	testArr, testLabelArr = loadDataSet(testFilename)
	predictionnumIt = adaClassify(testArr, classifierArray)

	errArr = np.mat(np.ones((len(testArr), 1)))
	errorRate = errArr[predictionnumIt != np.mat(testLabelArr).T].sum()

	return	errorRate












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