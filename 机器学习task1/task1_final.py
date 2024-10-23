import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#定义已有标签的Dataset类（不传入y）
class BostonHousingDataset(Dataset):
    def __init__(self,X,y):
        self.X=torch.tensor(X,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#加载波士顿房价数据集
data=pd.read_csv("train.csv")

#缺失值处理
imputer=SimpleImputer(strategy='mean')
data_imputed=imputer.fit_transform(data.drop('medv',axis=1))
data_imputed=pd.DataFrame(data_imputed,columns=data.columns[:-1])
data_imputed['medv']=data['medv']

#异常值处理
isolation_forest=IsolationForest(contamination=0.01)
data['anomaly']=isolation_forest.fit_predict(data_imputed)
data=data[data['anomaly']==1]
data=data.drop("anomaly",axis=1)#丢弃anomaly

#异常值处理后记得保存ID一列
data_ids=data['ID']

#划分数据集为训练集、验证集和测试集
X=data.drop(['medv',"ID"],axis=1)#丢弃ID和medv，防止进入回归之中
y=pd.DataFrame(data['medv'],columns=['medv'])
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#归一化处理，对特征进行归一化
scaler=MinMaxScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_val_scaled=scaler.transform(X_val)
X_test_scaled=scaler.transform(X_test)
#对y进行归一化
target_scaler=MinMaxScaler()
y_train_scaled=target_scaler.fit_transform(y_train)
y_val_scaled=target_scaler.fit_transform(y_val)
y_test_scaled=target_scaler.fit_transform(y_test)

#创建Dataset对象
train_dataset=BostonHousingDataset(X_train_scaled, y_train_scaled)
val_dataset=BostonHousingDataset(X_val_scaled, y_val_scaled)

#创建DataLoader对象
train_loader=DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader=DataLoader(val_dataset, batch_size=32, shuffle=False)

#定义神经网络模型
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1=nn.Linear(input_size,32)
        self.fc2=nn.Linear(32,10)
        self.fc3=nn.Linear(10,1)

    def forward(self, x):
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x

#初始化模型、损失函数和优化器
model=Net(input_size=X_train_scaled.shape[1])
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.005)

#训练模型：epoch选择50
epochs=50
epoch_list=[]
loss_list=[]
for epoch in range(epochs):
    model.train()
    for inputs,targets in train_loader:
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=criterion(outputs,targets)
        #反向传播
        loss.backward()
        optimizer.step()

    #验证模型，根据图像来调整epoch和lr，以及线性回归模型隐藏层参数
    model.eval()
    val_loss=0
    with torch.no_grad():
        for inputs,targets in val_loader:
            outputs=model(inputs)
            loss=criterion(outputs,targets)
            val_loss+=loss.item()
    #计算平均的loss
    avg_val_loss=val_loss/len(val_loader)
    print(f'Epoch {epoch+1}/{epochs}, Val Loss: {avg_val_loss:.4f}')
    #添加到list
    epoch_list.append(epoch)
    loss_list.append(avg_val_loss)
#画出损失函数
plt.plot(epoch_list, loss_list)
plt.title("BostonHousing Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

#使用train.csv分割得到的测试集评估模型，并且分区间评估预测的效果
model.eval()
#列表用于分类评估模型
bins=[5,10,15,20,25,30,35,np.inf]
labels=["5-10k","10-15k","15-20k","20-25k","25-30k","30-35k","35k+"]
mse_values=[]

test_loss=0
inputs=torch.tensor(X_test_scaled,dtype=torch.float32)
targets=torch.tensor(y_test_scaled,dtype=torch.float32)
with torch.no_grad():
    outputs=model(inputs)
    loss=criterion(outputs, targets)
    test_loss+=loss.item()

    #反归一化，用来评估每个类别的预测结构
    outputs=target_scaler.inverse_transform(outputs)
    targets=target_scaler.inverse_transform(targets)
    #使用bool掩码
    for i in range(len(bins) - 1):
        mask=(targets>= bins[i]) & (targets< bins[i + 1])  #mask是一个bool掩码,利用这种办法来分区间
        if np.any(mask):
            mse=mean_squared_error(targets[mask], outputs[mask])  #算出每个分类的mse
            mse_values.append(np.sqrt(mse/inputs.size(0)))
        else:
            mse_values.append(0)
#打印在评估中的平均误差
avg_test_loss=test_loss /inputs.size(0)
print(f"Test Loss: {avg_test_loss:.4f}")
#画出在每个区间的误差柱状图
plt.bar(labels,mse_values,label="MSE")
plt.xlabel("Price Range")
plt.ylabel("Error")
plt.title("BostonHousingModel")
plt.legend()
plt.show()



#加载test.csv文件
test_data=pd.read_csv("test.csv")
#开始处理数据
#丢弃ID一列，防止进入回归
test_X=test_data.drop("ID",axis=1)
#保存ID一列
test_data_ids=test_data["ID"]
#归一化处理：使用训练集的scaler
test_X_scaled=scaler.transform(test_X)
#传换成张量，相当于一个Dataset
test_X_tensor=torch.tensor(test_X_scaled, dtype=torch.float32)

#得到结构，连同ID一起保存在文件submission.csv之中
with torch.no_grad():
    model.eval()
    predictions=model(test_X_tensor)
    #反归一化处理，使用训练集target_scaler
    predictions=target_scaler.inverse_transform(predictions.detach().numpy())
    predicted_prices=predictions.flatten()  #需要展平才能添加到DataFrame

    #将预测结果保存到CSV文件
    predicted_prices_with_ids = pd.DataFrame({
    "ID": test_data_ids,
    "medv": predicted_prices
    })
    predicted_prices_with_ids.to_csv("submission.csv", index=False)