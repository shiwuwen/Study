

def binary_search(sortarray:list, target:int) -> int:
	high = len(sortarray)-1
	low = 0
	

	while(low<=high):
		#python // 运算符用于取整
		mid = (high + low) // 2
		print(mid)

		if sortarray[mid]>target:
			high = mid - 1
		elif sortarray[mid]<target:
			low = mid + 1
		else:
			return mid 

	return -1


if __name__ == '__main__':
	sortarray = [0, 3, 8, 19, 28, 56, 79, 109]

	result = binary_search(sortarray, 1)
	print(result)



