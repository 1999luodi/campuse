import torch
import torch.nn as nn
import torch.optim as optim
from dataset import data_load
from sklearn.metrics import accuracy_score 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

current_path=os.path.dirname(os.path.abspath(__file__))
# 假设我们有一个简单的分类模型
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim,hidden_dim=256):
        super(SimpleModel, self).__init__()
        
        self.model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_dim),
        nn.Dropout(0.5),  # Dropout 层
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_dim),
        nn.Dropout(0.5),  # Dropout 层
        nn.Linear(hidden_dim, output_dim)
        )
        self.apply(self.init_weights)
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将输入张量展平为 [batch_size, 3*128*128]
        return self.model(x)



# 训练函数
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def train(model, train_loader, criterion, optimizer, device):
    model.train()  # 设置为训练模式
    total_loss = 0.0
    all_labels = []
    all_preds = []

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据转移到设备
        labels = labels.to(torch.long)  # 确保 labels 是 int64 类型  

        optimizer.zero_grad()  # 梯度清零
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        total_loss += loss.item()

        # 获取预测值
        _, predicted = torch.max(outputs.data, 1)
        
        # 将当前批次的标签和预测值存储到列表中
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

    # 计算整个训练集的准确率
    accuracy = accuracy_score(all_labels, all_preds)
    
    # 计算其他评估指标
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # 打印指标
    # print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
    
    # 返回平均损失和准确率
    return total_loss / len(train_loader), accuracy, precision, recall, f1

def validate(model, val_loader, criterion, device):
    model.eval()  # 设置为评估模式
    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():  # 在验证过程中不计算梯度
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.to(torch.long)  # 确保 labels 是 int64 类型  

            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            total_loss += loss.item()

            # 获取预测值
            _, predicted = torch.max(outputs.data, 1)

            # 将当前批次的标签和预测值存储到列表中
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # 计算整体损失和准确率
    accuracy = accuracy_score(all_labels, all_preds)
    
    # 计算其他评估指标
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # 打印指标
    # print(f'Validation Loss: {total_loss / len(val_loader):.4f}')
    # print(f'Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
    
    return total_loss / len(val_loader), accuracy, precision, recall, f1

def test(model, test_loader, criterion, device):
    model.eval()  # 设置为评估模式
    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():  # 在测试过程中不计算梯度
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.to(torch.long)  # 确保 labels 是 int64 类型  

            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            total_loss += loss.item()

            # 获取预测值
            _, predicted = torch.max(outputs.data, 1)

            # 将当前批次的标签和预测值存储到列表中
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # 计算整体损失和准确率
    accuracy = accuracy_score(all_labels, all_preds)
    
    # 计算其他评估指标
    precision = precision_score(all_labels, all_preds, average='weighted',zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # # 打印指标
    # print(f'Test Loss: {total_loss / len(test_loader):.4f}')
    # print(f'Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
    
    return total_loss / len(test_loader), accuracy, precision, recall, f1


if __name__ == "__main__":
    
    # 数据加载
    train_loader, val_loader, test_loader = data_load()  # 假设 data_load 函数返回 DataLoader 对象

    # 设备设置（使用GPU或CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型、损失函数和优化器
    input_dim = 3 * 128 * 128  # 图像展开后的维度
    output_dim = 2  # 假设是二分类问题
    model = SimpleModel(input_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 使用Adam优化器

    # 使用 ReduceLROnPlateau 调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # 训练和验证
    num_epochs = 50
    validate_interval = 10  # 每 10 次训练迭代进行一次验证
    
    #模型best保存
    best_val_loss = float('inf')  # 初始化最佳验证损失为无穷大
    best_model_state = None  # 存储最佳模型的参数
    
    for epoch in range(num_epochs):
        # 训练阶段
        train_loss, train_accuracy, train_precision, train_recall, train_f1 = train(model, train_loader, criterion, optimizer, device)
        
       
         # 每10次训练才进行一次验证
        if (epoch + 1) % validate_interval == 0:
            # 验证阶段
            val_loss, val_accuracy, val_precision, val_recall, val_f1 = validate(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-score: {train_f1:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-score: {val_f1:.4f}")
            # 依据验证损失调整学习率
            scheduler.step(val_loss)  # 使用验证集损失来指导调度
            
            # 如果当前验证损失较低，保存模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()  # 保存当前模型的状态
                print(f"  Validation loss improved, saving model...")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-score: {train_f1:.4f}")
    
    # 在训练结束后，保存最佳模型
    if best_model_state is not None:
        path=os.path.join(current_path,'best_model.pth')
        torch.save(best_model_state, path)
        print("Best model saved.")
    
    # 测试阶段
    # 加载最好的模型
    model.load_state_dict(torch.load(path, weights_only=True))# 加载保存的最优模型权重
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = test(model, test_loader,criterion, device)
    print(f"  Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-score: {test_f1:.4f}")
