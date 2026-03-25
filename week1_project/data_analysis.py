import numpy as np
import pandas as pd

#1. 读取数据
df = pd.read_csv('E:\Pythod\week1_project\data.csv')
print("原始数据：")
print(df.head())

#2. 查看数据基本信息
print("\n数据基本信息：")
print(df.info())

#3. 数据清洗：处理缺失值
df = df.dropna()  # 删除含有缺失值的行
print("\n清洗后的数据：")
print(df)

#4. 数据统计分析计算
avg_score = np.mean(df['成绩'])
max_age = np.max(df['年龄'])
print(f"\n平均分数: {avg_score:.2f}")
print(f"最大年龄: {max_age}")

#5. 保存清洗后的数据
df.to_csv('cleaned_data.csv', index=False, encoding='utf-8')
print("\n清洗后的数据已保存到")

#=================函数封装：让代码更干净=================
def get_class_stats(data, class_name):
    """统计指定班级的成绩（函数定义+调用）"""
    class_data = data[data['班级'] == class_name]
    return np.mean(class_data["成绩"])
    
# 调用函数
class1_avg = get_class_stats(df, "一班")
print(f"\n一班的平均成绩: {class1_avg:.2f}")