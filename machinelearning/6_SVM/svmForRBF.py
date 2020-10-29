#coding=utf-8
import numpy as np


#简化版SMO实现SVM
def loadDataSet(filename):
	dataMat = []; labelMat = []
	fr = open(filename)
	for line in fr.readlines():
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]), float(lineArr[1])])
		labelMat.append(float(lineArr[2]))

	return dataMat, labelMat


def selectJrand(i, m):
	j = i
	while j==i:
		j = int(np.random.uniform(0, m))

	return j


def clipAlpha(aj, H, L):
	if aj > H:
		aj = H
	elif aj < L:
		aj = L
	
	return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
	dataMatrix = np.mat(dataMatIn)
	labelMat = np.mat(classLabels).transpose()
	b = 0 
	m, n = np.shape(dataMatrix) #m,2
	alphas = np.mat(np.zeros((m,1)))
	iter = 0

	while iter<maxIter:
		alphaPairsChanged = 0
		for i in range(m):
			fXi = float(np.multiply(alphas, labelMat).T*
				(dataMatrix * dataMatrix[i,:].T)) + b 
			Ei = fXi - float(labelMat[i])

			if((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or \
				((labelMat[i]*Ei>toler) and (alphas[i]>0)):
				j = selectJrand(i, m)
				fXj = float(np.multiply(alphas, labelMat).T*
					(dataMatrix * dataMatrix[j,:].T)) + b 
				Ej = fXj - float(labelMat[j])

				alphaIold = alphas[i].copy()
				alphaJold = alphas[j].copy()

				if labelMat[i] != labelMat[j]:
					L = max(0, alphas[j] - alphas[i])
					H = min(C, C + alphas[j] - alphas[i])
				else:
					L = max(0, alphas[j] + alphas[i] - C)
					H = min(C, alphas[j] + alphas[i])
				if L == H:
					print("L==H")
					continue

				eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T -\
						dataMatrix[i,:]*dataMatrix[i,:].T -\
						dataMatrix[j,:]*dataMatrix[j,:].T

				if eta>=0:
					print('eta>=0')
					continue

				alphas[j] -= labelMat[j]*(Ei - Ej)/eta
				alphas[j] = clipAlpha(alphas[j], H, L)

				if abs(alphas[j] - alphaJold) < 0.00001:
					print('j not moving enough')
					continue

				alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])

				b1 = b - Ei - labelMat[i] *(alphas[i]-alphaIold)*\
					dataMatrix[i,:]*dataMatrix[i,:].T -\
					labelMat[j]*(alphas[j]-alphaJold)*\
					dataMatrix[i,:]*dataMatrix[j,:].T

				b2 = b - Ej - labelMat[i] *(alphas[i]-alphaIold)*\
					dataMatrix[i,:]*dataMatrix[j,:].T -\
					labelMat[j]*(alphas[j]-alphaJold)*\
					dataMatrix[j,:]*dataMatrix[j,:].T

				if (alphas[i]>0) and (alphas[i]<C):
					b = b1
				elif (alphas[j]>0) and (alphas[j]<C):
					b = b2
				else:
					b = (b1+b2)/2.0

				alphaPairsChanged += 1
				print("iter: %d i: %d, pairs changed %d" %(iter, i, alphaPairsChanged))


		if alphaPairsChanged == 0:
			iter += 1
		else:
			iter = 0

		print('iteration number: %d' %iter)

	return b, alphas


def kernelTrans(X, A, kTup):
	m, n = np.shape(X)
	K = np.mat(np.zeros((m,1)))
	if kTup[0] == 'lin':
		K = X * A.T
	elif kTup[0] == 'rbf':
		for j in range(m):
			deltaRow = X[j,:] - A
			K[j] = deltaRow * deltaRow.T
		K = np.exp(K / (-1 * kTup[1]**2)) 
	else:
		print('kTup error')

	return K


class optStruct:
	def __init__(self, dataMatIn, classLabels, C, toler, kTup):
		self.X = dataMatIn
		self.labelMat = classLabels
		self.C = C
		self.tol = toler
		self.m = np.shape(dataMatIn)[0]
		self.alphas = np.mat(np.zeros((self.m, 1)))
		self.b = 0
		self.eCache = np.mat(np.zeros((self.m, 2)))
		self.K = np.mat(np.zeros((self.m, self.m)))
		for i in range(self.m):
			self.K[:,i] = kernelTrans(self.X, self.X[i, :], kTup)


def calcEk(oS, k):
	fXk = float(np.multiply(oS.alphas, oS.labelMat).T*oS.K[:,k]) + oS.b
	Ek = fXk - float(oS.labelMat[k])

	return Ek


def selectJ(i, oS, Ei):
	maxK = -1
	maxDeltaE = 0
	Ej = 0
	oS.eCache[i] = [1, Ei]
	# 对一个矩阵.A转换为Array类型
    # 返回误差不为0的数据的索引值
	vaildEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]

	if len(vaildEcacheList)>1:
		for k in vaildEcacheList:
			if k==i:
				continue
			Ek = calcEk(oS, k)
			deltaE = abs(Ei - Ek)
			if deltaE > maxDeltaE:
				maxK = k
				maxDeltaE = deltaE
				Ej = Ek
		return maxK, Ej
	else:
		j = selectJrand(i, oS.m)
		Ej = calcEk(oS, j)
	return j, Ej


def updateEk(oS, k):
	Ek = calcEk(oS, k)
	oS.eCache[k] = [1, Ek]


def innerL(i, oS):
	Ei = calcEk(oS, i)

	if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
		((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
		j, Ej = selectJ(i, oS, Ei)	
		alphaIold = oS.alphas[i].copy()
		alphaJold = oS.alphas[j].copy()

		if oS.labelMat[i] != oS.labelMat[j]:
			L = max(0, oS.alphas[j] - oS.alphas[i])
			H = min(oS.C, oS.alphas[j] - oS.alphas[i] + oS.C)	
		else:
			L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
			H = min(oS.C, oS.alphas[j] + oS.alphas[i])
		if L==H:
			print('L==H')

		eta = 2.0*oS.K[i,j] - oS.K[i,i] - oS.K[j,j]

		if eta>=0:
			print('eta>=0')
			return 0

		oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
		oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)

		updateEk(oS, j)

		if abs(oS.alphas[j] - alphaJold) < 0.00001:
			print('j not moving eough')
			return 0

		oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])

		updateEk(oS, i)

		b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*\
			oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j] - alphaJold)*\
			oS.K[i,j]

		b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*\
			oS.K[i,j] - oS.labelMat[j]*(oS.alphas[j] - alphaJold)*\
			oS.K[j,j]

		if (oS.alphas[i]>0) and (oS.alphas[i]<oS.C):
			oS.b = b1
		elif (oS.alphas[j]>0) and (oS.alphas[j]<oS.C):
			oS.b = b2
		else:
			oS.b = (b1 + b2)/2.0

		return 1

	else:
		return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
	oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
	iter = 0
	entirSet = True
	alphaPairsChanged = 0

	while (iter < maxIter) and ((alphaPairsChanged > 0) or entirSet):
		alphaPairsChanged = 0

		if entirSet:
			for i in range(oS.m):
				alphaPairsChanged += innerL(i, oS)
			print('fullset, iter: %d i: %d, pairs changed: %d' %(iter, i, alphaPairsChanged))
			iter += 1
		else:
			nonBoundIs = np.nonzero((oS.alphas.A>0) * (oS.alphas.A < C))[0]
			for i in nonBoundIs:
				alphaPairsChanged += innerL(i, oS)
				print('non-bound, iter: %d i: %d, pairs changed: %d' %(iter, i, alphaPairsChanged))
				iter += 1

		if entirSet:
			entirSet = False
		elif alphaPairsChanged == 0:
			entirSet = True

		print('iteration numbers: %d' %iter)

	return oS.b, oS.alphas


def calcWs(alphas, dataArr, classLabels):
	X = np.mat(dataArr)
	labelMat = np.mat(classLabels).T

	w = X.T * np.multiply(alphas, labelMat)

	return w


def testRbf(k1=1.3):
	dataArr, labelArr = loadDataSet('testSetRBF.txt')
	b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 100, ('rbf', k1))
	dataMat = np.mat(dataArr)
	labelMat = np.mat(labelArr).transpose()

	svInd = np.nonzero(alphas>0)[0]
	sVs = dataMat[svInd]
	labelSV = labelMat[svInd]

	print('there are %d support vectors' %np.shape(sVs)[0])

	m, n = np.shape(dataMat)
	errorCount = 0

	for i in range(m):
		kernelEval = kernelTrans(sVs, dataMat[i,:], ('rbf', k1))
		predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b

		if np.sign(predict) != np.sign(labelArr[i]):
			errorCount += 1
	print('训练集错误率:%.2f%%' % ((float(errorCount) / m) * 100))

	dataArr, labelArr = loadDataSet('testSetRBF2.txt')
	errorCount = 0
	datMat = np.mat(dataArr)
	labelMat = np.mat(labelArr).transpose()
	m, n = np.shape(datMat)
	for i in range(m):
		# 计算各个点的核
		kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
		# 根据支持向量的点计算超平面，返回预测结果
		predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
		# 返回数组中各元素的正负号，用1和-1表示，并统计错误个数
		if np.sign(predict) != np.sign(labelArr[i]):
			errorCount += 1
	# 打印错误率
	print('测试集错误率:%.2f%%' % ((float(errorCount) / m) * 100)) 






if __name__ == '__main__':
	testRbf()