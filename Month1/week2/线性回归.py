# 1. 导入库
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 2. 创建数据（学习时长 → 预测成绩）
data = {
    "学习时长": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "成绩": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
}
df = pd.DataFrame(data)

# 3. 划分特征 X 和标签 y
X = df[["学习时长"]]  # 输入：学习时间
y = df["成绩"]      # 输出：分数

# 4. 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. 创建线性回归模型
model = LinearRegression()

# 6. 训练
model.fit(X_train, y_train)

# 7. 评分
print("训练集分数：", model.score(X_train, y_train))
print("测试集分数：", model.score(X_test, y_test))

# 8. 预测：学习5小时，能考多少分？
print("预测成绩：", model.predict([[6]]))