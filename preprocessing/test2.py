import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

mask = Image.open('dataset/mask/112.png')
# mask = np.asarray(mask)
# print(mask)


# print(mask.shape)
# cv2.imshow('M', mask)
mask = transforms.Resize((512, 512))(mask)
# print(mask.shape)
# mask = cv2.imread('dataset/mask/112.png')
# mask = cv2.resize(mask, (512, 512))
# cv2.imshow('m', mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(mask.shape)
# print(mask[:, :, 2])
lb_arr = np.asarray([
    [0, 0, 0], # 0 for background
    [0, 255, 0], # 1 for plants
    [255, 0, 0]    # 2 for weeds
])

# mask = mask.convert('RGB')
mask = np.array(mask)

mask = mask.astype(int)
print(mask.shape)
label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
for ii, label in enumerate(lb_arr):
    label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
label_mask = label_mask.astype(int)

result = torch.tensor(label_mask)
print(result[result==2])
# image = Image.fromarray(label_mask)
# print(image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()