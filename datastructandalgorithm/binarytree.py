'''
树Tree是n个结点的有限集,n=0时称为空树。在任意一棵非空树中: 1、有且仅有一个特定的称为根Root的结
点；2、n>1时，其余结点可分为 (m>O) 个互不相交的有限集T1,...,Tm，其中每一个集合本身又是一槐树，并
且称为根的子树( SubTree )。

树的度：树中各节点度的最大值
结点的度(degree)：结点拥有的子树个数
叶节点(leaf)：度为0的结点
分支结点：度不为0的结点
内部结点：除根结点以外的分支结点

结点子树的根称为结点的孩子(child)；该结点称为孩子的双亲
同一双亲的孩子互称兄弟

树的深度等于树的高度，其中深度从根向子树计算，高度从叶子结点向根计算

树通过二叉链表的孩子兄弟表示法可以将一棵树转换成二叉树
'''

'''
二叉树：二叉树( Binary Tree)是n个结点的有限集合，该集合
或者为空集(称为空二叉树)，或着由一个根结点和两棵互不相交
的、分别称为根结点的左子树和右子树的二叉树组成

满二叉树：二叉树中，如果所有分支结点都存在左子树和右子树，并且所有叶子都
在同一层上，这样的二叉树称为满二叉树

完全二叉树：对一棵具有n个结点的二叉树按层序编号，如果编号为i(l<=i<=n) 的结点与同
样深度的满二叉树中编号为i的结点在二叉树中位置完全相同，则这棵二叉树称为完
全二叉树。

性质：
	1、二叉树的第i层上至多有2^(i-1)个节点(i>=1)
	2、深度为k的二叉树至多有(2^k)-1 个节点
	3、任一二叉树T，如果终端节点数为n0,度为二的结点数为n2，则n0=n2+1
		n=n0+n1+n2 n-1(分支数)=0*n0+1*n1+2*n2
	4、具有n各结点的完全二叉树的深度为floor(log2n)+1
'''

'''
哈夫曼树：
	将n个带权结点按从小到大的顺序排列
	从该序列中选取两个最小的结点，用一个新结点作为根将这两个结点变为一棵树，根结点的值为两个结点值之和，将该树依大小顺序放回原序列
	重复上述步骤，直到序列中只有一棵树为止，此时得到一棵哈夫曼树
'''

import queue
class TreeNode():

	def __init__(self, data=None, lchild=None, rchild=None):
		self.data = data
		self.lchild = lchild
		self.rchild = rchild

class BinaryTree():

	def __init__(self):
		pass

	def creattree(self, treequeue):
		'''
		使用前序遍历构造二叉树
		输入为扩展二叉树的前序遍历序列,使用$代替结点的空孩子
		'''
		curr = treequeue.get()

		if curr == '$':
			atree = None
			return atree
		else:
			atree = TreeNode()
			atree.data = curr
			atree.lchild = self.creattree(treequeue)
			atree.rchild = self.creattree(treequeue)
			return atree

	def preordertraverse(self, tree):
		if tree==None:
			# print('$', end=' ')
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

	def levelordertraverse(self, tree):
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

	'''
	树结构
						A 				
			B 						C
		D 		E  				F 		G
	H						I				J		
		K

	'''
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

	#从数组创建二叉树
	print()
	# treelist = ['A','B','$','D','$','$','C','$','$']
	treelist = ['A', 'B', 'D', 'H', '$', 'K', '$', '$', '$', 'E', '$', '$', 'C', 'F', 'I', '$', '$', '$', 'G', '$', 'J', '$', '$']
	treequeue = queue.Queue()
	#将数组转存至队列中
	for i in range(len(treelist)): 
		# print(treelist[i], end=' ')
		treequeue.put(treelist[i])
	print('创建二叉树：')
	atree = TreeNode()
	atree = btreemethod.creattree(treequeue)
	btreemethod.preordertraverse(atree)

	# print()
	# print('前序遍历：')
	# btreemethod.preordertraverse(A)

	# print()
	# print('中序遍历：')
	# btreemethod.inordertraverse(A)

	# print()
	# print('后序遍历：')
	# btreemethod.postordertraverse(A)

	# print()
	# print('层序遍历：')
	# btreemethod.levelordertraverse(A)

	#获取树的叶子结点
	# print()
	# print('叶子节点为：')
	# btreemethod.getleaves(A)

	#获取树的深度
	# print()
	# print('树的深度为：')
	# print(btreemethod.getdepth(A))

