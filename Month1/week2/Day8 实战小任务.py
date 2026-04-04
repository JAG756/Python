from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. 生成模拟数据
X, y = make_classification(n_samples=1000)

# 2. 划分训练集、测试集（8:2）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. 训练模型
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 4. 评分
print("训练集分数：", model.score(X_train, y_train))
print("测试集分数：", model.score(X_test, y_test))
