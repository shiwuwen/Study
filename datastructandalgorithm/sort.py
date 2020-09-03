

def swap(inarray:list, i:int, j:int):
	temp = inarray[i]
	inarray[i] = inarray[j]
	inarray[j] = temp

def simple_sort(inarray:list):
	nums = len(inarray)

	for i in range(nums):
		for j in range(i+1, nums):
			if inarray[i]>inarray[j]:
				swap(inarray, i, j)

def bubble_sort(inarray:list):
	nums = len(inarray)

	for i in range(nums-1):
		for j in range(nums-2, i-1, -1):
			if inarray[j]>inarray[j+1]:
				swap(inarray, j, j+1)
				print(inarray)

def select_sort(inarray:list):
	nums = len(inarray)

	for i in range(nums-1):
		index = i

		for j in range(i+1, nums):
			if inarray[index]>inarray[j]:
				index = j
		if index != i:
			swap(inarray, i, index)

def insert_sort(inarray:list):
	nums = len(inarray)

	for i in range(1, nums):
		temp = inarray[i]

		if inarray[i]< inarray[i-1]:
			j = i-1
			while inarray[j]>temp and j>-1:
				inarray[j+1] = inarray[j]
				j -= 1

			inarray[j+1] = temp

def quick_sort(inarray:list):
	nums = len(inarray)

	if nums<1:
		return inarray
	else:
		pivot = inarray[0]
		less = [i for i in inarray[1:] if i<=pivot]
		greater = [i for i in inarray[1:] if i>pivot]

		return quick_sort(less) + [pivot] + quick_sort(greater)


if __name__ == '__main__':
	a = [9, 1, 5, 8, 3, 7, 4, 6, 2]
	print(a)

	# simple_sort(a)

	# bubble_sort(a)

	# select_sort(a)

	# insert_sort(a)

	a = quick_sort(a)

	print(a)
