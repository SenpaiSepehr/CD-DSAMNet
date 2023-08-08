#coding=utf-8
from os import listdir
from os.path import join
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import numpy as np
import torchvision.transforms as transforms
import os

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png','.tif', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calMetric_iou(predict, label):
    tp = torch.sum((predict == 1) & (label == 1)).item()
    fp = torch.sum((predict == 1) & (label == 0)).item()
    fn = torch.sum((predict == 0) & (label == 1)).item()

    return tp,fp,fn

def predict_compressor(predict):

    """
    batch_size, num_channels, width, height = predict.shape
    new_predict = torch.zeros(batch_size, 1, width, height)

    for i in range(batch_size):
        for x in range(width):
            for y in range(height):

                ones_count = torch.sum(predict[i,:,x,y] == 1)
                zeros_count = num_channels - ones_count

                if ones_count > zeros_count:
                    new_predict[i,0,x,y] = 1
    """
    new_pred = (torch.sum(predict, dim=1) > 32).float().unsqueeze(1)

    return new_pred


def batch_fscore(predict, label):

    total_img_f1, batchf1 = 0,0

    for batch_id in range(predict.shape[0]):
        predict_img = predict[batch_id]
        label_img = label[batch_id]

        tp = torch.sum((predict_img == 1) & (label_img == 1)).item()+0.0001
        fp = torch.sum((predict_img == 1) & (label_img == 0)).item()+0.0001
        fn = torch.sum((predict_img == 0) & (label_img == 1)).item()+0.0001

        # f1-score of individual image-pairs
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        #print(tp,fp,fn)
        fscore = round((2 * (prec * rec) / (prec + rec)), 4)

        # saving total fscores
        total_img_f1 += fscore
    
    # batch f-score by averaging over batch size
    batchf1 = total_img_f1 / predict.shape[0]

    return batchf1


def batch_iou(predict, label):

    total_img_iou, batchIou = 0,0

    for batch_id in range(predict.shape[0]):
        predict_img = predict[batch_id]
        label_img = label[batch_id]

        # # operations in a 1D tensor are more efficient
        # predict_flat = predict_img.view(predict_img.shape[1], -1)
        # label_flat = label_img.view(label_img.shape[1], -1)

        # calculate individual image-pair iou
        tp = torch.sum((predict_img == 1) & (label_img == 1)).item()
        fp = torch.sum((predict_img == 1) & (label_img == 0)).item()
        fn = torch.sum((predict_img == 0) & (label_img == 1)).item()

        img_iou = round((tp / (tp+fp+fn)), 4)

        # accumulate all image iou
        total_img_iou += img_iou

    # batch iou by averaging over batch size
    batchIou = total_img_iou / 16

    return batchIou


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(64),
        CenterCrop(64),
        ToTensor()
    ])



def getSampleLabel(img_path):
    img_name = img_path.split('\\')[-1]
    return torch.from_numpy(np.array([int(img_name[0] == 'i')], dtype=np.float32))


def getDataList(img_path):
    dataline = open(img_path, 'r').readlines()
    datalist =[]
    for line in dataline:
        temp = line.strip('\n')
        datalist.append(temp)
    return datalist


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    return result


def get_transform(convert=True, normalize=False):
    transform_list = []
    if convert:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class LoadDatasetFromFolder(Dataset):
    def __init__(self, args, hr1_path, hr2_path, lab_path):
        super(LoadDatasetFromFolder, self).__init__()
        # 获取图片列表
        datalist = [name for name in os.listdir(hr1_path) for item in args.suffix if
                      os.path.splitext(name)[1] == item]

        self.hr1_filenames = [join(hr1_path, x) for x in datalist if is_image_file(x)]
        self.hr2_filenames = [join(hr2_path, x) for x in datalist if is_image_file(x)]
        self.lab_filenames = [join(lab_path, x) for x in datalist if is_image_file(x)]

        self.transform = get_transform(convert=True, normalize=True)  # convert to tensor and normalize to [-1,1]
        self.label_transform = get_transform()  # only convert to tensor

    def __getitem__(self, index):
        hr1_img = self.transform(Image.open(self.hr1_filenames[index]).convert('RGB'))
        hr2_img = self.transform(Image.open(self.hr2_filenames[index]).convert('RGB'))
        label = self.label_transform(Image.open(self.lab_filenames[index]))
        label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)

        return hr1_img, hr2_img, label

    def __len__(self):
        return len(self.hr1_filenames)
    
    
class TestDatasetFromFolder(Dataset):
    def __init__(self, Time1_dir, Time2_dir, Label_dir, image_sets):
        super(TestDatasetFromFolder, self).__init__()

        self.image1_filenames = [join(Time1_dir, x) for x in image_sets if is_image_file(x)]
        self.image2_filenames = [join(Time2_dir, x) for x in image_sets if is_image_file(x)]
        self.image3_filenames = [join(Label_dir, x) for x in image_sets if is_image_file(x)]

        self.transform = get_transform(convert=True, normalize=True)  # convert to tensor and normalize to [-1,1]
        self.label_transform = get_transform()

    def __getitem__(self, index):
        image1 = self.transform(Image.open(self.image1_filenames[index]).convert('RGB'))
        image2 = self.transform(Image.open(self.image2_filenames[index]).convert('RGB'))
        label = self.label_transform(Image.open(self.image3_filenames[index]))
        label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)

        image_name =  self.image1_filenames[index].split('/', -1)
        image_name = image_name[len(image_name)-1]

        return image1, image2, label, image_name

    def __len__(self):
        return len(self.image1_filenames)
