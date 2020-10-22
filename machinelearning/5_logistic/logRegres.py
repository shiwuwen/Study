import numpy as np


def loadDataSet():
	dataMat = []; labelMat = []
	fr = open('testSet.txt')
	for line in fr.readlines():
		lineArr = line.strip().split()
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	return dataMat, labelMat

def sigmod(inX):
	return 1.0 / (1+np.exp(-inX))

def gradAscent(dataMatIn, classLabel):
	'''
	batch gradient ascent
	批量梯度上升算法
	'''
	dataMatrix = np.mat(dataMatIn) #100*3
	labelMatrix = np.mat(classLabel).transpose() #100*1
	m, n = np.shape(dataMatrix)
	alpha = 0.001
	maxCycles = 500
	weights = np.ones((n,1)) #3*1

	for k in range(maxCycles):
		h = sigmod(dataMatrix*weights) #100*1
		error = labelMatrix - h #100*1
		weights = weights + alpha*dataMatrix.transpose()*error #3*1

		#getA()将矩阵转换为ndarray
	return weights.getA()


def stochasticGradAsent0(dataMatrix, classLabel):
	m, n = np.shape(dataMatrix)

	alpha = 0.01
	weights = np.ones(n) #3

	for i in range(m):
		h = sigmod(np.dot(dataMatrix[i], weights))
		error = classLabel[i] - h
		weights = weights + alpha*dataMatrix[i]*error

	return weights


def stochasticGradAsent1(dataMatrix, classLabel, numIter=150):
	m, n = np.shape(dataMatrix)

	# alpha = 0.01
	weights = np.ones(n) #3

	for j in range(numIter):
		dataIndex = list(range(m))

		for i in range(m):
			alpha = 4 / (1.0+j+i) + 0.01

			randIndex = int(np.random.uniform(0, len(dataIndex)))


			h = sigmod(np.dot(dataMatrix[randIndex], weights))
			error = classLabel[randIndex] - h
			weights = weights + alpha*dataMatrix[randIndex]*error
			del(dataIndex[randIndex])

	return weights

def plotBestFit(weights):
	import matplotlib.pyplot as plt 

	#将矩阵转换为ndarray
	# weights = wei.getA()

	dataMat, labelMat = loadDataSet()
	dataArr = np.array(dataMat)
	n = np.shape(dataArr)[0]

	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	for i in range(n):
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i,1])
			ycord1.append(dataArr[i,2])
		else:
			xcord2.append(dataArr[i,1])
			ycord2.append(dataArr[i,2])

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
	ax.scatter(xcord2, ycord2, s=30, c='green')
	x = np.arange(-3.0, 3.0, 0.1)
	y = (-weights[0] - weights[1]*x) / weights[2]
	ax.plot(x, y)
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.show()



#使用logistic回归从疝气病症预测病马的死亡率 
def classifyVector(inX, weights):
	prob = sigmod(sum(inX*weights))
	# prob = sigmod(np.dot(inX, weights))
	# print(prob)

	if prob > 0.5:
		return 1
	else:
		return 0


def colicTest():
	frTrain = open('horseColicTraining.txt')
	frTest = open('horseColicTest.txt')

	trainingSet = []; trainingLabels = []
	for line in frTrain.readlines():
		# \t 相当于tab键
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[21]))

	trainWeights = stochasticGradAsent1(np.array(trainingSet), trainingLabels, 500)

	errorCount = 0
	numTestVec = 0.0

	for line in frTest.readlines():
		numTestVec += 1
		currLine = line.strip().split('\t')
		lineArr = []

		for i in range(21):
			lineArr.append(float(currLine[i]))

		classifyRes = classifyVector(np.array(lineArr), trainWeights)
		# print(classifyRes, ' and ', currLine[21])
		if classifyRes != int(currLine[21]):
			errorCount += 1

	errorRate = float(errorCount) / numTestVec

	print('the error rate of this test is: ', errorRate)

	return errorRate


def multiTest():
	numTests = 10
	errorSum = 0.0

	for k in range(numTests):
		errorSum += colicTest()

	print('after %d iterations the average error rate is: %f' %(numTests, errorSum/float(numTests)))


if __name__ == '__main__':

	dataArr, labelMat = loadDataSet()
	# weights = gradAscent(dataArr, labelMat)
	# plotBestFit(weights)

	# weights2 = stochasticGradAsent1(np.array(dataArr), labelMat)
	# plotBestFit(weights2)

	multiTest()