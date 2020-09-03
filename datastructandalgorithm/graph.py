'''

'''
class GraphVertex():
	'''
	图的数据结构
	graphmatrix: 图的邻接矩阵
	vertex: 图的顶点列表
	nums: 顶点个数
	'''
	def __init__(self, graphmatrix:list, vertex:list, nums:int):
		self.graphmatrix = graphmatrix
		self.vertex = vertex
		self.nums = nums

def dfstraverse(graph:GraphVertex):
	'''
	深度优先遍历
	'''
	visited = [0 for i in range(graph.nums)]

	#处理非连通图的情况
	for i in range(graph.nums):
		if visited[i] == 0:
			dfs(graph, i, visited)

def dfs(graph:GraphVertex, i:int, visited:list):
	'''
	使用递归实现深度优先遍历
	'''
	visited[i] = 1
	print(graph.vertex[i], end=' ')

	for j in range(graph.nums):
		if visited[j] ==0 and graph.graphmatrix[i][j]==1:
			dfs(graph, j, visited)

import queue
def bfstraverse(graph:GraphVertex):
	'''
	广度优先遍历
	使用队列实现
	'''
	que = queue.Queue()
	visited = [0 for i in range(graph.nums)]

	if graph.nums > 0:
		que.put(0)
		visited[0] = 1
	else:
		return 'error'

	while not que.empty():
		index = que.get()
		print(graph.vertex[index], end=' ')

		for j in range(nums):
			if visited[j]==0 and graph.graphmatrix[index][j]>0:
				que.put(j)
				visited[j] = 1
import copy
def dijkstra(graph:GraphVertex, index:int) -> list:
	'''
	Dijkstra算法
	用于计算单源最短路径
	适用于不含负权边的图
	'''
	visited = [0 for i in range(graph.nums)]
	path = [index for i in range(graph.nums)]
	
	#若不使用copy.deepcopy，将修改原数组graph.graphmatrix的内容
	leastcost = copy.deepcopy(graph.graphmatrix[index])
	for i in range(graph.nums):
		if leastcost[i]==0:
			leastcost[i] = 65536

	# print(leastcost)

	visited[index] = 1

	for i in range(graph.nums):
		min = 65536
		k = -1
		for j in range(graph.nums):
			if visited[j]==0 and leastcost[j]<min:
				min = leastcost[j]
				k = j

		visited[k] = 1
		for l in range(graph.nums):
			if visited[l]==0 and graph.graphmatrix[k][l]>0 and leastcost[k]+graph.graphmatrix[k][l]<leastcost[l]:
				leastcost[l] = leastcost[k]+graph.graphmatrix[k][l]
				path[l] = k

	return leastcost, path

def getpath(path:list, index:int, i:int):
	'''
	显示Dijkstra算法生成的路径
	'''

	j = path[i]
	print(i, '--', end=' ')
	while j != index:
		print(j, '--' , end=' ')
		j = path[j]
	print(index)
	
def minispantree_prim(graph):
	'''
	最小生成树，prim算法
	使用贪心算法计算
	'''
	visited = [0 for i in range(graph.nums)]
	path = [0 for i in range(graph.nums)]
	visited[0] = 1
	lowcost = copy.deepcopy(graph.graphmatrix[0])
	for i in range(graph.nums):
		if lowcost[i]==0:
			lowcost[i] = 65536

	for i in range(graph.nums):
		min = 65536
		k = -1
		for j in range(graph.nums):
			if visited[j]==0 and lowcost[j]<min:
				min = lowcost[j]
				k = j 

		visited[k] = 1
		for l in range(graph.nums):
			if visited[l]==0 and graph.graphmatrix[k][l] !=0 and graph.graphmatrix[k][l]<lowcost[l]:
				lowcost[l] = graph.graphmatrix[k][l]
				path[l] = k

	return path, lowcost


if __name__ == '__main__':
	vertex = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
	graphmatrix = [
		   #  A,  B,  C,  D,  E,  F,  G,  H,  I
			[ 0, 10,  0,   0,  0, 11,  0,  0,  0],
			[10,  0, 18,  0,  0,  0, 16,  0, 12],
			[ 0, 18,  0, 22,  0,  0,  0,  0,  8],
			[ 0,  0, 22,  0, 20,  0, 24, 16, 21],
			[ 0,  0,  0, 20,  0, 26,  0,  7,  0],
			[11,  0,  0,  0, 26,  0, 17,  0,  0],
			[ 0, 16,  0, 24,  0, 17,  0, 19,  0],
			[ 0,  0,  0, 16,  7,  0, 19,  0,  0],
			[ 0, 12,  8, 21,  0,  0,  0,  0,  0]
			]	

	nums = 9

	graph = GraphVertex(graphmatrix, vertex, nums)

	# #深度优先搜索
	# print('深度优先搜索： ')
	# dfstraverse(graph)
	# print()

	# #广度优先搜索
	# print('广度优先搜索： ')
	# bfstraverse(graph)
	# print()

	#dijkstra单源最短路径
	leastcost, path = dijkstra(graph, 0)
	print(graph.graphmatrix[0])
	print(leastcost)
	print(path)
	for i in range(nums):
		getpath(path, 0, i)

	# #最小生成树 prim算法
	# path, lowcost = minispantree_prim(graph)
	# print(path)
	# print(lowcost)
	# for i in range(1,nums):
	# 	print('({},{})'.format(i,path[i]))

	