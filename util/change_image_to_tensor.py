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
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)
    return loader

def loader_dataset():
    train_dataset_path = "./dataset/same_patch_dataset_1000/left/"
    label_dataset_path = './dataset/same_patch_dataset_1000/right/'

    transform_train = transforms.Compose([transforms.Resize((64,64)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.468,0.432,0.393],std=[0.265,0.255,0.258])
                                  ])
    transform_label = transforms.Compose([transforms.Resize((64,64)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.445,0.410,0.378],std=[0.281,0.270,0.270])
                                  ])
    train_loader = change_image_to_batch_tensor(train_dataset_path, transform_train)
    label_loader = change_image_to_batch_tensor(label_dataset_path, transform_label)
    return train_loader, label_loader

if __name__ == "__main__":
    train_loader, label_loader = loader_dataset()
    for (mini_train_data, mini_label_data) in zip(train_loader, label_loader):
        print("mini_train:"+str(mini_train_data.shape))
        print("mini_label:"+str(mini_label_data.shape))

