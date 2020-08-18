'''
树
'''


'''
						A 				
			B 						C
		D 		E  				F 		G
	H						I				J		
		K

'''
import queue
class TreeNode():

	def __init__(self, data, lchild=None, rchild=None):
		self.data = data
		self.lchild = lchild
		self.rchild = rchild

class BinaryTree():

	def __init__(self):
		pass

	def preordertraverse(self, tree):
		if tree==None:
			return
		print(tree.data, end=' ')

		self.preordertraverse(tree.lchild)
		self.preordertraverse(tree.rchild)

	def inordertraverse(self, tree):
		if tree==None:
			return
		self.inordertraverse(tree.lchild)
		print(tree.data, end=' ')
		self.inordertraverse(tree.rchild)

	def postordertraverse(self, tree):
		if tree==None:
			return
		self.postordertraverse(tree.lchild)
		self.postordertraverse(tree.rchild)
		print(tree.data, end=' ')

	def levelorder(self, tree):
		que = queue.Queue()
		curr = tree 

		if curr == None:
			print('empty tree')
		else:
			que.put(curr)
			while not que.empty():
				curr = que.get()
				print(curr.data, end=' ')
				if curr.lchild is not None:
					que.put(curr.lchild)
				if curr.rchild is not None:
					que.put(curr.rchild)

	def getleaves(self, tree):
		if tree == None:
			return

		self.getleaves(tree.lchild)
		self.getleaves(tree.rchild)

		if tree.lchild==None and tree.rchild==None:
			print(tree.data, end=' ')

	
	def getdepth(self, tree):
		'''
		规定只有一个节点的树的深度为1
		'''
		if tree == None:
			return 0
		elif tree.lchild==None and tree.rchild==None:
			return 1
		elif tree.lchild==None and tree.rchild!=None:
			return 1 + self.getdepth(tree.rchild)
		elif tree.lchild!=None and tree.rchild==None:
			return 1 + self.getdepth(tree.lchild)
		elif tree.lchild!=None and tree.rchild!=None:
			return 1 + max(self.getdepth(tree.rchild), self.getdepth(tree.lchild))



if __name__ == '__main__':
	K = TreeNode('K')
	H = TreeNode('H', rchild=K)
	D = TreeNode('D', lchild=H)
	E = TreeNode('E')
	B = TreeNode('B', lchild=D, rchild=E)

	I = TreeNode('I')
	F = TreeNode('F', lchild=I)
	J = TreeNode('J')
	G = TreeNode('G', rchild=J)
	C = TreeNode('C', lchild=F, rchild=G)

	A = TreeNode('A', lchild=B, rchild=C)

	btreemethod = BinaryTree()

	# print('前序遍历：')
	# btreemethod.preordertraverse(A)

	# print()
	# print('中序遍历：')
	# btreemethod.inordertraverse(A)

	# print()
	# print('后序遍历：')
	# btreemethod.postordertraverse(A)

	print()
	print('层序遍历：')
	btreemethod.levelorder(A)

	# print()
	# print('叶子节点为：')
	# btreemethod.getleaves(A)

	# print()
	# print('树的深度为：')
	# print(btreemethod.getdepth(A))
