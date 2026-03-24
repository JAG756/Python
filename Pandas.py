import pandas as pd
import numpy as np

# ======================
# 1. 创建示例数据（也可以换成读取你的文件）
# ======================
data = {
    "姓名": ["张三", "李四", "王五", "赵六", "钱七", "李四", None],
    "年龄": [22, 25, 30, np.nan, 28, 25, 32],
    "城市": ["北京", "上海", "广州", "深圳", "北京", "上海", "上海"],
    "工资": [5000, 6000, 8000, 7500, 5500, 6000, 7000]
}

df = pd.DataFrame(data)
print("原始数据：")
print(df)
print("-" * 50)

# ======================
# 2. 读取外部文件（替换上面的创建数据即可）
# ======================
# 读取 CSV
# df = pd.read_csv("你的文件.csv")

# 读取 Excel
# df = pd.read_excel("你的文件.xlsx")

# 读取 TXT
# df = pd.read_csv("你的文件.txt", sep="\t")

# ======================
# 3. 数据清洗（最常用）
# ======================

# 查看基本信息
print("数据基本信息：")
df.info()
print("-" * 50)

# 查看缺失值
print("每列缺失值数量：")
print(df.isnull().sum())
print("-" * 50)

# ① 删除缺失值
df = df.dropna()

# ② 删除重复行
df = df.drop_duplicates()

# ③ 重置索引（清洗后必备）
df = df.reset_index(drop=True)

print("清洗后数据：")
print(df)
print("-" * 50)

# ======================
# 4. 简单统计分析
# ======================

# 描述性统计（均值、标准差、最大/最小、分位数）
print("描述性统计：")
print(df.describe())
print("-" * 50)

# 单列统计
print("平均工资：", df["工资"].mean())
print("最高工资：", df["工资"].max())
print("年龄总和：", df["年龄"].sum())
print("数据行数：", len(df))
print("-" * 50)

# 分组统计（按城市统计平均工资）
print("按城市统计平均工资：")
print(df.groupby("城市")["工资"].mean())
print("-" * 50)

# 计数统计（每个城市多少人）
print("各城市人数：")
print(df["城市"].value_counts())