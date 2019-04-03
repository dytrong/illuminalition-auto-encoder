import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os

class Descriptor_Dataset(Dataset):
    """
    root: 图像存放地址根路径
    augment: 图像增强
    """
    def __init__(self, root, augment=None):
        # 这个list存放所有图像的地址
        self.image_files = np.array([x.path for x in os.scandir(root) if x.name.endswith(".jpg") or x.name.endswith(".png")])
        self.augment = augment

    def __getitem__(self, index):
        if self.augment:
            image = Image.open(self.image_files[index]).convert('RGB')
            image = self.augment(image)
            return image
        else:
            return Image.open(self.image_files[index]).convert('RGB')

    def __len__(self):
        # 返回图像数量
        return len(self.image_files)


def change_image_to_batch_tensor(dataset_path, transform=None):
    dataset = Descriptor_Dataset(dataset_path, augment=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)
    return loader

def loader_dataset():
    train_dataset_path = "./dataset/same_patch_dataset_1000/left/"
    label_dataset_path = './dataset/same_patch_dataset_1000/right/'

    transform = transforms.Compose([transforms.Resize((64,64)),
                                    transforms.ToTensor()
                                  ])
    train_loader = change_image_to_batch_tensor(train_dataset_path, transform)
    label_loader = change_image_to_batch_tensor(label_dataset_path, transform)
    return train_loader, label_loader

def compute_tensor_mean_std(dataloader):
    tensor_mean = []
    tensor_std = []

    for i, mini_data in enumerate(dataloader):
        # shape (batch_size, 3, height, width)
        numpy_image = mini_data.numpy()
        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std = np.std(numpy_image, axis=(0, 2, 3))
        # 写入list中
        tensor_mean.append(batch_mean)
        tensor_std.append(batch_std)
    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    tensor_mean = np.array(tensor_mean).mean(axis=0)
    tensor_std = np.array(tensor_std).mean(axis=0)
    print(tensor_mean)
    print(tensor_std)
    return tensor_mean, tensor_std

if __name__ == '__main__':
    _, dataloader= loader_dataset()
    tensor_mean, tensor_std = compute_tensor_mean_std(dataloader)
      
