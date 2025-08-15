from keras.models import Sequential

from keras.layers import Dense ,Activation 

model = Sequential()


#完成模型搭建
#将一些网络层通过 .add（）堆叠起来，就构成了模型
model.add(Dense(uint=64,input_dim=100))#64个神经元，100个输入：  # units：神经元数量；input_dim：输入特征维度
model.add(Activation("relu"))# 激活函数：ReLU（修正线性单元）
model.add(Dense(uint=10))## 第2个全连接层：10个神经元（无input_dim，自动匹配上层输出）
model.add(Activation("softmax")) # 激活函数：Softmax（多分类概率归一化）


#调用.compile（）来编译模型
model.compile(loss="categorical_crossentropy",optimizer="rmsprop",metrics=["accuracy"])

#编译模型的时候必须指明损失函数和优化器
#损失函数：categorical_crossentropy # 损失函数：多分类交叉熵（适用于独热编码标签）
#优化器：rmsprop：# 优化器：RMSprop（自适应学习率优化器）
#指标：accuracy： # 评估指标：准确率（分类正确样本占比）



#有需要的话也可以自己制定损失函数

# from Keras.optimizers import SGD#随机梯度下降优化器
# model.compile(loss="categorical_crossentropy",optimizer=SGD(lr=0.1,momentum=0.9,nesterov=True))
## lr：学习率；momentum：动量；nesterov：是否启用Nesterov动量

#训练模型：
#完成编译后，我们可以在训练数据上按batch进行一定次数的迭代来训练网络
model.fit(
    x_train, y_train,          # x_train：训练特征数据；y_train：训练标签数据
    epochs=5,                  # 训练轮次：整个数据集的迭代次数（代码中`s`为未定义变量，此处修正为示例值5）
    batch_size=32              # 批次大小：每次梯度更新的样本数（通常取2的幂次如16/32/64）
)

#当然也可以进行手动一个一个将bath的数据送入网络
#model.train_on_batch(x_train,y_train)

# 评估模型性能
loss_and_metrics = model.evaluate(
    x_test, y_test,            # x_test：测试特征数据；y_test：测试标签数据（代码中`x_text`为笔误，应修正为`x_test`）
    batch_size=128             # 评估批次大小：一次性载入的测试样本数
)

# 预测新数据
classes = model.predict(
    x_test,                    # 待预测的输入数据
    batch_size=128             # 预测批次大小
)





