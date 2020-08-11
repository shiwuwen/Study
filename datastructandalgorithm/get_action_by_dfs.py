# import sys
# sys.setrecursionlimit(100000)

import copy
def dfs(nums, sum, actions, tmp):
	
	if nums < 1:
		print('error: num should greater than 1')

	elif nums == 1:
		tmp.append(sum/10)
		
		node = copy.deepcopy(tmp)
		# print(node)
		actions.append(node)
		# print(actions)
		tmp.pop()
		# print(actions)
		# return actions
	else:
		# print('no run')
		for i in range(sum+1):
			tmp.append(i/10)
			dfs(nums-1, sum-i, actions, tmp)
			tmp.pop()
	
	# return actions 



actions = []
tmp = []
for i in range(1,5):
	dfs(i, 10, actions, tmp)

	print('%d个服务器时共有 %d 种可能'%(i,len(actions)))
	print(actions)
	actions = []
