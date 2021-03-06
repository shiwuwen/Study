#coding=utf-8
import numpy as np


#简化版SMO实现SVM
def loadDataSet(filename):
	'''
	从filename中加载数据
	'''
	#数据和标签列表
	dataMat = []; labelMat = []
	fr = open(filename)
	for line in fr.readlines():
		#以tab为分隔符
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]), float(lineArr[1])])
		labelMat.append(float(lineArr[2]))

	return dataMat, labelMat


def selectJrand(i, m):
	'''
	随机选择一个与i不同的下标j
	'''
	j = i
	#直到i,j不相等
	while j==i:
		j = int(np.random.uniform(0, m))

	return j


def clipAlpha(aj, H, L):
	'''
	将aj的值限制在[L,H]之间
	'''
	if aj > H:
		aj = H
	elif aj < L:
		aj = L
	
	return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
	'''
	SMO的简单实现
	'''
	#将输入转换成矩阵 m,2
	dataMatrix = np.mat(dataMatIn)
	#将标签转换为矩阵并转置 m,1
	labelMat = np.mat(classLabels).transpose()
	#偏移b
	b = 0 
	#数据集大小
	m, n = np.shape(dataMatrix) #m,2
	#初始化权重为0 m,1
	alphas = np.mat(np.zeros((m,1)))
	#当前迭代次数
	iter = 0

	while iter<maxIter:
		#当前改变的alpha数
		alphaPairsChanged = 0
		#遍历每一个样本
		for i in range(m):
			#预测当前样本的标签
			fXi = float(np.multiply(alphas, labelMat).T*
				(dataMatrix * dataMatrix[i,:].T)) + b 
			#计算预测与实际值的差
			Ei = fXi - float(labelMat[i])

			#如果Ei在误差toler之外，则优化alpha
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

				#更新alphaj
				alphas[j] -= labelMat[j]*(Ei - Ej)/eta
				alphas[j] = clipAlpha(alphas[j], H, L)

				if abs(alphas[j] - alphaJold) < 0.00001:
					print('j not moving enough')
					continue

				alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])

				#更新b1
				b1 = b - Ei - labelMat[i] *(alphas[i]-alphaIold)*\
					dataMatrix[i,:]*dataMatrix[i,:].T -\
					labelMat[j]*(alphas[j]-alphaJold)*\
					dataMatrix[i,:]*dataMatrix[j,:].T

				#更新b2
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


# def kernelTrans(X, A, kTup):
# 	'''
# 	for RBF
# 	'''
# 	m, n = np.shape(X)
# 	K = np.mat(np.zeros((m,1)))
# 	if kTup[0] == 'lin':
# 		K = X * A.T
# 	elif kTup[0] == 'rbf':
# 		for j in range(m):
# 			deltaRow = X[j,:] - A
# 			K[j] = deltaRow * deltaRow.T
# 		K = exp(K / (-1 * kTup[1]**2)) 
# 	else:
# 		print('kTup error')

# 	return K

#完整SMO算法
class optStruct:
	def __init__(self, dataMatIn, classLabels, C, toler):
		self.X = dataMatIn
		self.labelMat = classLabels
		self.C = C
		self.tol = toler
		self.m = np.shape(dataMatIn)[0]
		self.alphas = np.mat(np.zeros((self.m, 1)))
		self.b = 0
		self.eCache = np.mat(np.zeros((self.m, 2)))
		# self.K = np.mat(np.zeros((self.m, self.m)))
		# for i in range(self.m):
		# 	self.K[:,i] = kernelTrans(self.X, self.X[i, :], kTup)


def calcEk(oS, k):
	fXk = float(np.multiply(oS.alphas, oS.labelMat).T*
			(oS.X*oS.X[k,:].T)) + oS.b
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

		eta = 2.0*oS.X[i,:]*oS.X[i,:].T - oS.X[i,:]*oS.X[i,:].T\
				- oS.X[j,:]*oS.X[j,:].T
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
			oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j] - alphaJold)*\
			oS.X[i,:]*oS.X[j,:].T

		b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*\
			oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j] - alphaJold)*\
			oS.X[j,:]*oS.X[j,:].T

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
	oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)
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


def test(w, b, xIn):
	res = np.mat(xIn) * w + b

	return res






if __name__ == '__main__':
	dataArr, labelArr = loadDataSet('testSet.txt')
	# print(labelArr)

	# b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
	b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
	print(b)
	print(alphas[alphas>0])

	w = calcWs(alphas, dataArr, labelArr)
	print('w', w)
	# print(type(w))
	print(test(w, b , dataArr[0]))
	print(labelArr[0])
