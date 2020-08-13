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