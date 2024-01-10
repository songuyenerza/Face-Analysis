import numbers
import os
import queue as Queue
import threading
import json
import cv2
import random


import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import albumentations as A
from PIL import Image, ImageEnhance


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


def random_crop(image, ratio = 0.07):
    pct_focusx = random.uniform(0, ratio)
    pct_focusy = random.uniform(0, ratio)
    x, y = image.size
    image = image.crop((x*pct_focusx, y*pct_focusy, x*(1-pct_focusx), y*(1-pct_focusy)))
    
    return image

class FaceDataset(Dataset):
    def __init__(self, root_dir, json_path, dict_class, target_size=(112, 112)):
        super(FaceDataset, self).__init__()
        #   augment

        # Custom padding transform
        self.padding_transform = transforms.Lambda(lambda img: self.pad_to_square(img))

        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
            self.padding_transform,
            transforms.ColorJitter(brightness=0.2, contrast=0.2,saturation=0.15, hue=0.15 ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Resize((120, 120)),
            transforms.RandomCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.RandomErasing(scale=(0.02, 0.1))
            ])

        self.transform_albument = A.Compose([
            A.Blur(blur_limit=(3, 7), p=0.15),
            A.Defocus(radius=(1, 2), p=0.15),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            # A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.1, p=0.1),              # have bug at this
            A.RingingOvershoot(blur_limit=(5, 13), p=0.15),
            A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=False, p=0.1),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.1),
            A.RandomBrightness(limit=0.2, always_apply=False, p=0.2),
            A.ImageCompression(quality_lower=20, quality_upper=100, p=0.2),
        ])

        self.dict_class = dict_class
        self.root_dir = root_dir
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def pad_to_square(self, pil_img):
        width, height = pil_img.size
        max_wh = max(width, height)
        hp = int((max_wh - width) / 2)
        vp = int((max_wh - height) / 2)
        padding = (hp, vp, hp, vp)  # left, top, right, bottom
        # For grayscale padding value (128, 128, 128) or normalized (0.5, 0.5, 0.5)
        padding_color = (128, 128, 128)
        return transforms.functional.pad(pil_img, padding, padding_color, 'constant')

    def __getitem__(self, index):
        data_item = self.data[str(index)]
        img = cv2.imread(os.path.join(self.root_dir, data_item['path']))

        # end augment
        sample = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        #   todo: sonnt
        #   augment with albumentation
        transformed = self.transform_albument(image=sample)
        sample = transformed["image"]
        #   //////////////////////////

        label = int(self.dict_class[data_item['labels']])
        label = torch.tensor(label, dtype=torch.long)
        if self.transform is not None:
            sample = self.transform(sample)
        sample = torch.tensor(np.asarray(sample))
        return sample, label

    def __len__(self):
        return len(self.data)

