# ======================
# Day10-2：决策树 分类
# 直接运行！
# ======================
from sklearn.datasets import load_iris
#自带的数据集，鸢尾花数据集
from sklearn.model_selection import train_test_split
#把数据集划分为训练集和测试集
from sklearn.tree import DecisionTreeClassifier
#决策树模型
from sklearn.metrics import accuracy_score
#算准确率

# 1. 加载数据
iris = load_iris()
#加载鸢尾花数据
X = iris.data
#花的尺寸数据（花瓣长、花瓣宽、花萼长、花萼宽）
y = iris.target
#花的类别（0、1、2分别代表三种不同的鸢尾花）

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#把数据集划分为训练集和测试集，测试集占20%，随机种子设为42以保证结果可复现

# 3. 创建决策树模型
tree = DecisionTreeClassifier(random_state=42)
#创建一个决策树，名字叫tree
#决策树 = 像流程图一样判断：“如果花瓣长> 3 → 是 A 类花”

# 4. 训练
tree.fit(X_train, y_train)
#用训练集的数据来训练决策树模型，让它学会如何根据花的尺寸来判断花的类别

# 5. 预测
y_pred = tree.predict(X_test)
#用训练好的决策树模型来预测测试集的花的类别，得到预测结果y_pred

# 6. 评分
print("===== 决策树 结果 =====")
print(f"预测准确率: {accuracy_score(y_test, y_pred):.2f}")
print(f"模型分数: {tree.score(X_test, y_test):.2f}")
#用accuracy_score函数计算预测结果的准确率，输出结果保留两位小数
#tree.score()方法也可以直接计算模型在测试集上的准确率，结果应该和accuracy_score一样

#这段代码做了一件事：
#拿鸢尾花数据（X = 尺寸，y = 品种）
#分成学习数据、考试数据
#用决策树学习规律
#让模型考试
#输出正确率