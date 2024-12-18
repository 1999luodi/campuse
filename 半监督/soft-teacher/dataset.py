import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn import preprocessing

# 定义Dataset类
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        """
        :param image_paths: 图片路径列表
        :param labels: 图片对应的标签列表
        :param transform: 图像预处理函数
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # 读取图片

        if self.transform:
            image = self.transform(image)  # 图像预处理

        label = self.labels[idx] if self.labels is not None else None
        return image, label

# 图像预处理（例如，缩放，归一化等）
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整大小
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

def prepare_data(data_dir):
    """
    加载数据集并为每个类别生成标签
    :param data_dir: 数据集的根目录，包含train, val, test
    :return: 返回处理好的训练、验证和测试数据
    """
    # 创建文件路径列表和标签
    classes = ['dog', 'cat']
    label_encoder = preprocessing.LabelEncoder()
    # 生成映射标签
    label_encoder.fit(classes)
    
    # 打印类别标签的编码映射关系
    print("标签编码映射：")
    for i, label in enumerate(label_encoder.classes_):
        print(f"{label} -> {label_encoder.transform([label])[0]}")

    def load_images_from_folder(folder):
        image_paths = []
        labels = []
        for class_name in os.listdir(folder):
            class_folder = os.path.join(folder, class_name)
            if os.path.isdir(class_folder):
                for filename in os.listdir(class_folder):
                    image_paths.append(os.path.join(class_folder, filename))
                    
                    labels.append(label_encoder.transform([class_name])[0])  # 将类别转换为数字标签
        return image_paths, labels

    # 加载训练集、验证集和测试集
    train_images, train_labels = load_images_from_folder(os.path.join(data_dir, 'train'))
    val_images, val_labels = load_images_from_folder(os.path.join(data_dir, 'val'))
    test_images, test_labels = load_images_from_folder(os.path.join(data_dir, 'test'))

    return train_images, train_labels, val_images, val_labels, test_images, test_labels

def data_load(batch_size=4):
    # 数据集目录
    data_dir = 'data/dataset'  # 这里替换为你的数据集路径
    
    # 准备数据
    train_images, train_labels, val_images, val_labels, test_images, test_labels = prepare_data(data_dir)

    # 创建Dataset实例
    
    train_dataset = ImageDataset(train_images, labels=train_labels, transform=transform)
    val_dataset = ImageDataset(val_images, labels=val_labels, transform=transform)
    test_dataset = ImageDataset(test_images, labels=test_labels, transform=transform)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,drop_last=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True, num_workers=2)
    
    # 示例: 遍历并打印每个批次的图片和标签形状
    print("Training data:")
    for images, labels in train_loader:
        print(f'Images batch shape: {images.shape}, Labels batch shape: {labels.shape}')

    print("\nValidation data:")
    for images, labels in val_loader:
        print(f'Images batch shape: {images.shape}, Labels batch shape: {labels.shape}')

    print("\nTesting data:")
    for images, labels in test_loader:
        print(f'Images batch shape: {images.shape}, Labels batch shape: {labels.shape}')
    return train_loader,val_loader,test_loader

if __name__ == "__main__":
    data_load()
