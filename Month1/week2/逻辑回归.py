# 1. 导入库
import pandas as pd
#导入做表格的工具，叫做pandas，习惯上我们把它叫pd，用来处理数据
from sklearn.linear_model import LogisticRegression
#导入逻辑回归模型，用来做分类问题
from sklearn.model_selection import train_test_split
#拆分数据集的工具，可以把数据分成训练集和测试集

# 2. 数据：学习时长 → 是否及格
data = {
    "学习时长": [1,2,3,4,5,6,7,8],
    "是否及格": [0,0,0,0,1,1,1,1]  # 0=不及格 1=及格
}

df = pd.DataFrame(data)
#把数据变成一张 Excel 表格，名字叫 df
#创建一个DataFrame，包含两列：学习时长和是否及格    

# 3. 划分 X 和 y
X = df[["学习时长"]]
#X 是特征数据，也就是用来预测的输入数据，这里是学习时长
y = df["是否及格"]
#y 是目标数据，也就是我们要预测的输出数据，这里是是否及格

# 4. 训练测试拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#把数据集拆分成训练集和测试集，测试集占20%，默认随机种子会导致每次运行结果不同

# 5. 逻辑回归模型
model = LogisticRegression()
#创建一个二分类模型（逻辑回归模型），名字叫model
model.fit(X_train, y_train)
#用训练集的数据来训练模型，让它学会根据学习时长来判断是否及格

# 6. 评分
print("训练分数：", model.score(X_train, y_train))
#模型在训练集上的表现，分数越高越好
print("测试分数：", model.score(X_test, y_test))
#模型在测试集上的表现，分数越高越好，过拟合的模型在训练集上分数很高，但在测试集上分数很低

# 7. 预测：学习5小时，会不会及格？
print("预测结果（0=不及格，1=及格）：", model.predict([[5]]))
#一个人学习了5小时，模型预测他会不会及格，输出结果是0或1，0代表不及格，1代表及格