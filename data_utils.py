from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

import numpy as np

import pdb

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.pgm'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        ToPILImage(),
        # Resize((120,120), interpolation=Image.BICUBIC),
        # ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        # Resize((30,30), interpolation=Image.BICUBIC),
        # ToTensor()
    ])

def val_hr_transform():
    return Compose([
        ToPILImage(),
        # Resize((120,120), interpolation=Image.BICUBIC),
    ])


def val_lr_transform():
    return Compose([
        Resize((30,30), interpolation=Image.BICUBIC),
    ])




def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        # CenterCrop(400),
        ToTensor()
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        ori_image = Image.open(self.image_filenames[index])
        w, h = ori_image.size

        ori_image = np.array(ori_image, dtype=np.uint8)
        ori_image = np.array([ori_image for i in range(3)]).transpose(1,2,0)
        hr_image = self.hr_transform(ori_image)

        lr_resize = Resize((h//4,w//4), interpolation=Image.BICUBIC)
        lr_image = lr_resize(hr_image)

        return ToTensor()(lr_image), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

        self.image_hr_transform = val_hr_transform()
        self.image_lr_transform = val_lr_transform()

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        # change to 3 channal
        hr_image = np.array(hr_image, dtype=np.uint8)
        hr_image = np.array([hr_image for i in range(3)]).transpose(1,2,0)

        hr_image = self.image_hr_transform(hr_image)

        lr_resize = Resize((h//4,w//4), interpolation=Image.BICUBIC)
        lr_image = lr_resize(hr_image)

        

        # lr_image = self.image_lr_transform(hr_image)
        restore_size = Resize((h,w), interpolation=Image.BICUBIC)
        hr_restore_img = restore_size(lr_image)

        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/LR'
        self.hr_path = dataset_dir + '/HD'

        self.image_hr_transform = val_hr_transform()

        # self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.hr_filenames[index].split('/')[-1]
        # lr_image = Image.open(self.lr_filenames[index])
        # # hr_image = Image.open(self.hr_filenames[index])

        # lr_image = np.array(lr_image, dtype=np.uint8)
        # lr_image = np.array([lr_image for i in range(3)]).transpose(1,2,0)
        # lr_image = self.image_hr_transform(lr_image)



        # hr_image = np.array(hr_image, dtype=np.uint8)
        # hr_image = np.array([hr_image for i in range(3)]).transpose(1,2,0)
        # hr_image = self.image_hr_transform(hr_image)

        # w, h = lr_image.size
        # restore_size = Resize((4*h, 4*w), interpolation=Image.BICUBIC)
        
        # # hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        # hr_restore_img = restore_size(lr_image)
        # hr_image = restore_size(hr_image)
        hr_image = Image.open(self.hr_filenames[index])
        w, h = hr_image.size
        # change to 3 channal
        hr_image = np.array(hr_image, dtype=np.uint8)
        hr_image = np.array([hr_image for i in range(3)]).transpose(1,2,0)

        hr_image = self.image_hr_transform(hr_image)

        lr_resize = Resize((h//4,w//4), interpolation=Image.BICUBIC)
        lr_image = lr_resize(hr_image)

        

        # lr_image = self.image_lr_transform(hr_image)
        restore_size = Resize((h,w), interpolation=Image.BICUBIC)
        hr_restore_img = restore_size(lr_image)

        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

        # return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
