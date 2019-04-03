# illuminalition-auto-encoder
This project is using auto encoder to solve illumilation changes


### data:20190109
### author:daizhuang

### 1.提取特征点 
python generate_patch_daset.py

### 2.根据ground true提取实际特征点匹配对 
python select_same_patches.py

### 3.计算数据集的均值和标准差
python compute_dataset_mean_std.py

### 4.将数据集转化为pytorch格式的tensor,并利用多线程读取数据
python change_image_to_tensor.py

### 5.训练autoencoder模型
python train_cnn_autoencoder.py

### 6.提取autoencoder的encoder特征，作为特征点描述符
调用autoencoder_descriptors.py函数，提取描述符
