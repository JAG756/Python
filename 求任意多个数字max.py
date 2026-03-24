def get_max(*args):
    if not args:
        return None
    max_value = args[0]
    for num in args:
        if num > max_value:
            max_value = num
    return max_value
# Example usage:
print(get_max(3, 1, 4, 1, 5, 9))  # Output: 9print(get_max())  # Output: None
print(get_max(-1, -5, -3))  # Output: -1        

def print_user(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

    print(f"共收到{len(kwargs)}个信息")

# Example usage:
print_user(name="Alice", age=30, city="New York")
# Output:
# name: Alice
# age: 30
# city: New York
# 共收到3个信息


