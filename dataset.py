from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import config
from torchvision.utils import save_image
class MapDataset(Dataset):
    def __init__(self,root_dir) -> None:
        super().__init__()
        self.rootdir = root_dir
        self.list_files = os.listdir(self.rootdir)
        print(self.list_files)

    def __len__(self):
        return len(self.list_files)

    # 因为数据集中的图片是两个连接起来的
    # 我们需要把两个拆开来看
    def __getitem__(self, index)  :
        img_file = self.list_files[index]#其实是单独的文件名
        img_path = os.path.join(self.rootdir,img_file)#这里是图片的路径
        image = np.array(Image.open(img_path))
        input_image = image[:,:600,:]#左边的原始图像
        target_image = image[:,600:,:]#右边的目标图像

        augmentations = config.both_transform(image = input_image,image0 = target_image)#缩小到256*256
        input_image, target_image = augmentations['image'],augmentations['image0']

        input_image = config.transform_only_input(image = input_image)["image"]
        target_image = config.transform_only_mask(image = target_image)["image"]

        return input_image,target_image
# if __name__ == "__main__":
#     dataset = MapDataset(config.TRAIN_DIR)
#     loader = DataLoader(dataset, batch_size=1)
#     for x, y in loader:
#         print(x.shape)
#         save_image(x, "x.png")
#         save_image(y, "y.png")
#         import sys

#         sys.exit()