'''
线性表：0个或多个数据元素的有序序列
线性表的顺序存储结构：
	用一段地址连续的存储单元依次存储线性表的数据元素
	插入删除时间复杂度：O(n)
线性表的链式存储结构：
	使用节点Node存储数据，节点包含数据域和指针域
	插入删除时间复杂度O(1)
循环链表：
	将最后一个元素的next指针指向head
双向链表：
	添加prior指针指向上一节点
	出入顺序：s.prior = p; s.next = p.next; p.next.prior = s; p.next = s	
'''

'''
栈：仅在表尾进行插入和删除操作的线性表
栈的顺序存储结构：
	两栈共享空间：两个栈共享一个数组，一个在头一个在尾，两栈互补关系
栈的链式储存结构：

中缀表达式转后缀表达式：
	从左到右遍历中缀表达式的每个数字和符号，若是数字就输出，即成为后
	缀表达式的一部分。若是符号，则判断其与栈顶符号的优先级，是右括号或优先
	级低于栈顶符号(乘除优先加减)则将顶元素依次出钱并输出，并将当前符号进栈，
	一直到最终输出后缀表达式为止。
后缀表达式计算：
	从左到右遍历后缀表达式，遇到数字就进栈，遇到符号就将栈顶两个数字出栈并将
	计算结果入栈，直到运算结束
'''

'''
队列：只允许在一端进行插入操作，而在另一端进行删除操作的线性表
循环队列：头尾相接的顺序存储结构
	front指向队列头，rear指向队列尾（队尾不存储数据）
	队列空则 front==rear
	队列满则 (rear+1)%QueueSize==front
	队列长度 (rear-front+QueueSize)%QueueSize
	QueueSize包含不存储数据的位置
'''



class Node():
	def __init__(self, data, next=None):
		self.data = data
		self.next = next

class LinkedList():
	'''
	单链表实现方法
	'''

	def __init__(self, head:Node=Node(-1)):
		self.__head = head

	def isempty(self):
		return self.__head.next == None

	def addelement(self, value):
		'''
		末尾添加元素
		'''
		p = self.__head

		q = Node(value)

		while p.next!=None:
			p = p.next
		p.next = q
		q.next = None
		# if p.next == None:
		# 	p.next = elem
		# 	elem.next = None
		# else:
		# 	q = p.next
		# 	elem.next = q
		# 	p.next = elem

	def showlist(self) -> list:
		'''
		返回list
		'''
		p = self.__head
		result = []
		if p.next == None:
			return result
		else:
			while p.next != None:
				q = p.next
				result.append(q.data)
				p = q
			return result

	def getelement(self, index:int) -> int or str:
		'''
		获取index处value值
		'''
		p = self.__head.next

		j = 1

		while p!=None and j<index :
			p = p.next
			j += 1 

		if p==None or j>index:

			return 'error'

		return p.data

	def insertelement(self, value, index:int) -> str:
		'''
		index 前插入
		'''
		p = self.__head
		node = Node(value)

		j = 1

		while p.next!=None and j<index:
			p = p.next
			j += 1

		if p.next==None or j>index:
			return 'error'
		else:
			q = p.next
			node.next = q
			p.next = node
			return 'OK'

	def deletenode(self, index:int) -> int or str:

		p = self.__head

		j = 1

		while p!=None and j<index:
			p = p.next
			j += 1

		if p.next==None or j>index:
			return 'error'
		else:
			q = p.next
			p.next = q.next
			return q.data





if __name__ == '__main__':

	lista = LinkedList()
	print(lista.isempty())
	lista.addelement(1)
	lista.addelement(2)
	# lista.addelement(c)
	print(lista.showlist())
	print(lista.insertelement(3,1))
	print(lista.getelement(3))
	print(lista.showlist())
	print(lista.deletenode(2))
	print(lista.showlist())

