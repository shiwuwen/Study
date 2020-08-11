class Node():
	def __init__(self, data, next=None):
		self.data = data
		self.next = next

class LinkedList():

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

