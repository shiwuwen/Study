'''
串（字符串）：由0个或多个字符组成的有限序列
串的比较：
	给定两个串，s="a1,a2,...,an" t="b1,b2,...,bm",当满足一下条件之一时，s<t
	1、n<m，且ai=bi(i=1,...,n)
	2、存在某个k<min(n,m),使得ai=bi(i=1,...,k-1),ak<bk
串的朴素匹配法：
	将两字符串逐字符比较
	最坏时间复杂度：O((n-m+1)*m)
KMP模式匹配法：
	寻找前缀与后缀集合的最长匹配项长度
'''


def naivestringmatching(S, T):
	j = 0
	i = 0

	while i<len(S) and j<len(T):

		if S[i]==T[j]:
			i += 1
			j += 1
		else:
			i = i-j+1
			j = 0
	# print(j)
	if j>len(T)-1:
		return i-len(T)
	else:
		return -1

def get_next(T):
	next = [-1 for i in range(len(T))]
	i = 0
	j = -1
	next[0] = -1

	while i<len(T)-1:
		if j==-1 or T[i]==T[j]:
			i += 1
			j += 1 
			next[i] = j
		else:
			j = next[j]

	return next

def get_nextval(T):
	nextval = [-1 for i in range(len(T))]
	i = 0
	j = -1
	nextval[0] = -1

	while i<len(T)-1:
		
		if j==-1 or T[i]==T[j]:
			i += 1
			j += 1
			if T[i]!=T[j]:
				nextval[i] = j
			else:
				nextval[i] = nextval[j]
		else:
			j = nextval[j]

	return nextval


def kmp(S, T):
	i = 0
	j = 0

	next = get_next(T)

	while i<len(S) and j<len(T):
		print(i,j)
		if j==-1 or S[i]==T[j]:
			i += 1 
			j += 1
		else:
			j = next[j]

	if j>len(T)-1:
		return i-len(T)
	else:
		return -1


if __name__ == '__main__':
	# s = 'abcdefghui'
	# t = 'hui'
	# print(len(s))
	# print(naivestringmatching(s,t))

	s = 'abababca'
	t = 'aaaba'
	print(get_next(s))
	# print(kmp(s,t))