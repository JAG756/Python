import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(arr[0])  # 输出第一个元素
print(arr[1:4])  # 输出第二到第四个元素
arr[0] = 10  # 修改第一个元素
print(arr)  # 输出修改后的数组

arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2[0, 1])  # 输出第一行第二列的元素
print(arr2[1, :])  # 输出第二行的所有元素
print(arr2[:, 2])  # 输出第三列的所有元素
print(arr2[0:2, 0:2])  # 输出第一行和第二行的前两列元素
arr2[0, 0] = 10  # 修改第一行第一列的元素
print(arr2)  # 输出修改后的二维数组

arr3 = np.array([1, 2, 3, 4, 5])
print(arr3 + 10)  # 输出每个元素加10的结果
print(arr3 * 2)  # 输出每个元素乘以2的结果

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)  # 输出两个数组对应元素相加的结果
print(a * b)  # 输出两个数组对应元素相乘的结果
print(a.dot(b))  # 输出两个数组的点积结果

arr4 = np.array([[1, 2, 3], [4, 5, 6]])

print(np.sum(arr4))  # 输出数组所有元素的和
print(np.mean(arr4))  # 输出数组所有元素的平均值
print(np.max(arr4))  # 输出数组所有元素的最大值
print(np.min(arr4))  # 输出数组所有元素的最小值

print(np.sum(arr4, axis=0))  # 输出每列的和
print(np.sum(arr4, axis=1))  # 输出每行的和

c = np.array([[1, 2], [3, 4]])
d = np.array([[5, 6], [7, 8]])

print(c + d)  # 输出两个二维数组对应元素相加的结果
print(c * d)  # 输出两个二维数组对应元素相乘的结果
print(c.dot(d))  # 输出两个二维数组的点积结果