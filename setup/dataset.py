from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import torch
import numpy as np
from PIL import Image
import os

class WeedDataset(Dataset):
    def __init__(self, root, random_transform=True):
        self.root = root
        self.random_transform = random_transform
        self.random_transform = {'hflip': TF.hflip,
                                'vflip': TF.vflip,
                                'rotate': TF.rotate}
        self.lb_arr = np.asarray([
            [0, 0, 0], # 0 for background
            [0, 255, 0], # 1 for plants
            [255, 0, 0]    # 2 for weeds
        ])

    def __len__(self):
        return len(os.listdir(os.path.join(self.root, 'mask')))

    def __getitem__(self, index):
        rgb_name = os.path.join(self.root, 'rgb', str(index)+'.png')
        nir_name = os.path.join(self.root, 'nir', str(index)+'.png')
        mask_name = os.path.join(self.root, 'mask', str(index)+'.png')


        rgb = Image.open(rgb_name).convert('RGB')
        nir = Image.open(nir_name)
        mask = Image.open(mask_name)

        nir =nir.crop((171, 93, 1060, 592))

        rgb = transforms.Resize((512, 512))(rgb)
        nir = transforms.Resize((512, 512))(nir)
        mask = transforms.Resize((512, 512))(mask)

        r,g,b = rgb.split()
        image = Image.merge('RGBA', (r,g,b,nir))
        image = transforms.Resize((512, 512))(image)

        if self.random_transform==True:
            image, mask, rgb = self._random_transform(image, mask, rgb)

        mask = np.array(mask)
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.lb_arr):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        result = torch.tensor(label_mask)

        image = TF.to_tensor(image)
        rgb = TF.to_tensor(rgb)

        sample = {'index': int(index), 'image': image, 'mask': result, 'rgb': rgb}
        return sample


    def _random_transform(self, image, mask, rgb):
        choice_list = list(self.random_transform)
        for _ in range(len(choice_list)):
            choice_key = random.choice(choice_list)

            action_prob = random.randint(0, 1)
            if action_prob >= 0.5:
                if choice_key == 'rotate':
                    rotation = random.randint(15, 75)
                    image = self.random_transform[choice_key](image, rotation)
                    mask = self.random_transform[choice_key](mask, rotation)
                    rgb = self.random_transform[choice_key](rgb, rotation)
                else:
                    image = self.random_transform[choice_key](image)
                    mask = self.random_transform[choice_key](mask)
                    rgb = self.random_transform[choice_key](rgb)
            choice_list.remove(choice_key)

        return image, mask, rgb
