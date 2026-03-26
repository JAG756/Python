# ======================
# Day10-2：决策树 分类
# 直接运行！
# ======================
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 创建决策树模型
tree = DecisionTreeClassifier(random_state=42)

# 4. 训练
tree.fit(X_train, y_train)

# 5. 预测
y_pred = tree.predict(X_test)

# 6. 评分
print("===== 决策树 结果 =====")
print(f"预测准确率: {accuracy_score(y_test, y_pred):.2f}")
print(f"模型分数: {tree.score(X_test, y_test):.2f}")