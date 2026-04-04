fruits = ["apple", "banana", "orange", "grape"]
print(fruits[0])  # 输出: apple
print(fruits[1])  # 输出: banana
print(fruits[2])  # 输出: orange
print(fruits[3])  # 输出: grape
print(fruits[-1])  # 输出: grape
print(len(fruits))  # 输出: 4
fruits.append("pear")  # 添加一个元素
print(fruits)  # 输出: ['apple', 'banana', 'orange', 'grape', 'pear']
fruits.remove("banana")  # 移除一个元素
print(fruits)  # 输出: ['apple', 'orange', 'grape', 'pear']
fruits[1] = "kiwi"  # 修改一个元素
print(fruits)  # 输出: ['apple', 'kiwi', 'grape', 'pear']
fruits.insert(1, "melon")  # 在指定位置插入一个元素
print(fruits)  # 输出: ['apple', 'melon', 'kiwi', 'grape', 'pear']
del fruits[2]  # 删除一个元素
print(fruits)  # 输出: ['apple', 'melon', 'grape', 'pear']

for f in fruits:
    print("我爱吃:", f)
# 输出:
# apple 
# melon
# grape
# pear

students = {
    "name": "jelf",
    "age": 22,
    "grade": 85,
    "city": "Nanjing"
}
print(students["name"])  # 输出: jelf
print(students["age"])   # 输出: 22
print(students["grade"]) # 输出: 85
print(students["city"])  # 输出: Nanjing
students["grade"] = 90  # 修改一个键的值    
print(students)  # 输出: {'name': 'jelf', 'age': 22, 'grade': 90, 'city': 'Nanjing'}
students["hobby"] = "coding"  # 添加一个新的键值对
print(students)  # 输出: {'name': 'jelf', 'age': 22, 'grade': 90, 'city': 'Nanjing', 'hobby': 'coding'}
del students["city"]  # 删除一个键值对
print(students)  # 输出: {'name': 'jelf', 'age': 22, 'grade': 90, 'hobby': 'coding'}
for key in students:
    print(key, ":", students[key])
# 输出:
# name : jelf
# age : 22
# grade : 90
# hobby : coding
for key, value in students.items():
    print(key, ":", value)
# 输出:
# name : jelf   
# age : 22
# grade : 90
# hobby : coding

def say_hello(name):
    return "Hello, " + name + "!"
greeting = say_hello("Alice")
print(greeting)  # 输出: Hello, Alice!
print(say_hello("Bob"))  # 输出: Hello, Bob!


def add_numbers(a, b):
    return a + b
result = add_numbers(5, 3)
print(result)  # 输出: 8

def introduce(name, age):
    return "My name is " + name + " and I am " + str(age) + " years old."
print(introduce("jelf", 22))  
# 输出: My name is jelf and I am 22 years old.
introduce("Alice", 30)
# 输出: My name is Alice and I am 30 years old.

with open("data.txt", "w") as file:
    file.write("Hello, this is a test file.\n")
    file.write("This file is used to demonstrate file handling in Python.\n")

with open("data.txt", "r") as file:
    content = file.read()
    print(content)
# 输出:
# Hello, this is a test file.   
# This file is used to demonstrate file handling in Python.

with open("data.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        print(line.strip())
# 输出:
# Hello, this is a test file.   
# This file is used to demonstrate file handling in Python.

with open("data.txt", "a") as file:
    file.write("This line is appended to the file.\n")
with open("data.txt", "r") as file:
    content = file.read()
    print(content)
# 输出:
# Hello, this is a test file.
# This file is used to demonstrate file handling in Python.
# This line is appended to the file.


foods = ["pizza", "burger", "sushi"]

myself = {
    "name": "jelf", 
    "age": 22,
    "hobby": "coding"
}

def multiply(x, y):
    return x * y  

with open("info.txt", "w") as file:
    file.write("这是一个包含列表、字典、函数和文件操作的示例。\n")
    file.write("我最喜欢的食物: " + str(foods) + "\n")
    file.write("我的信息: " + str(myself) + "\n")
    file.write("函数示例: multiply(5, 3) = " + str(multiply(5, 3)) + "\n")  

with open("info.txt", "r") as file:
    content = file.read()
    print(content)
# 输出:
# 这是一个包含列表、字典、函数和文件操作的示例。
# 我最喜欢的食物: ['pizza', 'burger', 'sushi']
# 我的信息: {'name': 'jelf', 'age': 22, 'hobby': 'coding'}
# 函数示例: multiply(5, 3) = 15

def add_numbers(*args):
    total = 0
    for num in args:
        total += num
    return total

result = add_numbers(1, 2, 3, 4, 5)
print(result)  # 输出: 15
print(add_numbers(10, 20))  # 输出: 30
print(add_numbers())  # 输出: 0
    
def print_info(**kwargs):
    print(kwargs)
    for key, value in kwargs.items():
        print(key + ":", value)

print_info(name="jelf", age=22, hobby="coding")
# 输出:
# {'name': 'jelf', 'age': 22, 'hobby': 'coding'}
# name: jelf
# age: 22
# hobby: coding

def func(*args, **kwargs):
    print("位置参数:", args)
    print("关键字参数:", kwargs)
func(1, 2, 3, name="jelf", age=22)
# 输出:
# 位置参数: (1, 2, 3)
# 关键字参数: {'name': 'jelf', 'age': 22}

def avg(*nums):
    return sum(nums) / len(nums) 
print(avg(1, 2, 3, 4, 5))  # 输出: 3.0
print(avg(10, 20, 30))  # 输出: 20.0

def send_msg(msg, **kwargs):
    print("消息内容:", msg)
    if "level" in kwargs:
        print("消息级别:", kwargs["level"])
    if "time" in kwargs:
        print("发送时间:", kwargs["time"])

send_msg("这是一个测试消息", level="info", time="2024-06-01 10:00:00")
# 输出:
# 消息内容: 这是一个测试消息
# 消息级别: info
# 发送时间: 2024-06-01 10:00:00

send_msg("你好")
# 输出:
# 消息内容: 你好

def print_user(name, age, city):
    print(name, age, city)

    user_list = ["Alice", 30, "New York"]
    print_user(*user_list)
# 输出:
# Alice 30 New York

user_dict = {"name": "Bob", "age": 25, "city": "Los Angeles"}
print_user(**user_dict)
# 输出:
# Bob 25 Los Angeles


