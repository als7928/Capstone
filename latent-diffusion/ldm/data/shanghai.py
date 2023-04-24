import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ShanghaiBase(Dataset):
    def __init__(self,
                 data_dir,
                 size=None,
                 interpolation="bicubic"
                #  flip_p=0.5
                 ):
        # self.data_paths = txt_file
        self.data_dir = data_dir
        self.image_paths = []
        for subdir, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".png"):
                    self.image_paths.append(os.path.join(subdir, file))
        # with open(self.data_paths, "r") as f:
        #     self.image_paths = f.read().splitlines()



        self._length = len(self.image_paths)
        # self.labels = {
        #     "relative_file_path_": [l for l in self.image_paths],
        #     "file_path_": [os.path.join(self.data_dir, l)
        #                    for l in self.image_paths],
        # }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        # self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        image = Image.open(image_path)

        # example = dict((k, self.labels[k][i]) for k in self.labels)
        # image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        # image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        # example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        # return example
        return image


class ShanhaiTrain(ShanghaiBase):
    def __init__(self, **kwargs):
        super().__init__(data_dir="train/train_data/train_img", **kwargs)


class ShanhaiValidation(ShanghaiBase):
    def __init__(self, **kwargs):
        super().__init__(data_dir="train/train_data/test_img",**kwargs)