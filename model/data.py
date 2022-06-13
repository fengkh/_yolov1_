from torch.utils.data import Dataset
import os
from PIL import Image
import torch


class MyDataset(Dataset):
    def __init__(self, root_dir, img_dir, label_dir):
        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, img_dir)
        self.label_dir = os.path.join(self.root_dir, label_dir)
        self.img_filename = os.listdir(self.img_dir)
        self.label_filename = os.listdir(self.label_dir)
        self.boxs = []
        self.labels = []
        self.dataset_type = 'txt'
        # 一次性读入所有的label,txt的字符串中包含了类别和标注框坐标，映射成list:[[label,x_center,y_center,w,h],[],...]
        if self.dataset_type == 'txt':
            for i in range(self.__len__()):
                with open(os.path.join(self.label_dir, self.label_filename[i]), 'r') as f:
                    txt = f.readlines()
                    f.close()
                self.txt2boxs_labels(txt)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.img_filename[idx]))
        target = self.encoder(self.boxs[idx], self.labels[idx])
        return image, target

    def __len__(self):
        return len(self.img_filename)

    # 把txt格式的标签转换成labels和boxs的格式，boxs为[[x_left_top,y_left_top,x_right_bottom,y_right_bottom],[],...]
    def txt2boxs_labels(self, txt):
        boxs = []
        labels = []
        for i in range(len(txt)):
            labels.append(int(txt[i].split(' ')[0]))
            x_center = float(txt[i].split(' ')[1])
            y_center = float(txt[i].split(' ')[2])
            w = float(txt[i].split(' ')[3])
            h = float(txt[i].split(' ')[4].rstrip())
            x_lt = x_center - w / 2
            y_lt = y_center - h / 2
            x_rb = x_center + w / 2
            y_rb = y_center + h / 2
            boxs.append([x_lt, y_lt, x_rb, y_rb])
        self.boxs.append(torch.Tensor(boxs))
        self.labels.append(torch.Tensor(labels))

    # 把标签和边框坐标编码成（7,7,30）的tensor
    def encoder(self, boxs, labels):
        grid_num = 7
        grid_size = 1.0 / grid_num
        params = 30
        target = torch.zeros((grid_num, grid_num, params))
        wh = boxs[:, 2:] - boxs[:, :2]
        xy_center = (boxs[:, 2:] + boxs[:, :2]) / 2
        for i in range(xy_center.size()[0]):
            center_point = xy_center[i]
            ij = (center_point * grid_num).floor()
            target[int(ij[1]), int(ij[0]), 4] = 1  # confidence1 = 1
            target[int(ij[1]), int(ij[0]), 9] = 1  # confidence2 = 1
            target[int(ij[1]), int(ij[0]), int(labels[i]) + 9] = 1  # encode label
            x1y1 = ij * grid_size
            delta_xy = (center_point - x1y1) / grid_size
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
            target[int(ij[1]), int(ij[0]), :2] = delta_xy
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
        return target


# MAIN = True
# if __name__ == '__main__' and MAIN:
#     dataset = MyDataset('E:\code_python\_yolov1_\dataset', 'img', 'label')
#     img, target = dataset.__getitem__(3)
#     print(target)
