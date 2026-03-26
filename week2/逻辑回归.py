# 1. 导入库
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 2. 数据：学习时长 → 是否及格
data = {
    "学习时长": [1,2,3,4,5,6,7,8],
    "是否及格": [0,0,0,0,1,1,1,1]  # 0=不及格 1=及格
}
df = pd.DataFrame(data)

# 3. 划分 X 和 y
X = df[["学习时长"]]
y = df["是否及格"]

# 4. 训练测试拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. 逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. 评分
print("训练分数：", model.score(X_train, y_train))
print("测试分数：", model.score(X_test, y_test))

# 7. 预测：学习5小时，会不会及格？
print("预测结果（0=不及格，1=及格）：", model.predict([[5]]))