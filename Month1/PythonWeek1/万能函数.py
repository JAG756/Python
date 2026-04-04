def log(func):
    def wrapper(*args, **kwargs):
        print("函数开始执行")
        result = func(*args, **kwargs)
        print("函数执行结束")
        return result
    return wrapper

@log
def add(a, b):
    return a + b

@log
def say(name, msg):
    print(name, "说", msg)

print(add(3, 5))
say("Alice", "Hello, World!")




