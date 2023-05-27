import torch
import numpy as np
import PIL
from PIL import Image
from scripts.sample_diffusion import load_model
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from einops import rearrange
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

class ShanghaiTestBase(Dataset):
    def __init__(self,
                 input_img,
                 size=None,
                 interpolation="bicubic"
                 ):
        self._length = 1
        self.img = input_img
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.cond_paths = [0]
        self. labels = {
            "cond_path_": [l for l in self.cond_paths],
        }
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = self.img
        if not image.mode == "RGB": 
            image = image.convert("RGB")    

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        # crop = min(img.shape[0], img.shape[1])
        # h, w, = img.shape[0], img.shape[1]
        # img = img[(h - crop) // 2:(h + crop) // 2,
        #     (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = np.array(image).astype(np.uint8)
        
        example['rgb'] = (image / 127.5 - 1.0).astype(np.float32)
            
        return example

class ShanghaiTest(ShanghaiTestBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

def ldm_cond_sample(config_path, ckpt_path, batch_size, out_name):
    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # x = iter(dataloader)
    x = next(iter(dataloader))
    seg = x['rgb']
    # seg = imageloader(image_name, size=512)
    # convert_tensor = transforms.ToTensor()
    # convert_tensor(seg)
    # seg = image
    with torch.no_grad():
        seg = rearrange(seg, 'b h w c -> b c h w')
        condition = model.to_rgb(seg)

        seg = seg.to('cuda').float()
        seg = model.get_learned_conditioning(seg)

        samples, _ = model.sample_log(cond=seg, batch_size=batch_size, ddim=True,
                                      ddim_steps=200, eta=1.)

        samples = model.decode_first_stage(samples)

    save_image(condition, out_name + '_cond.png')
    save_image(samples, out_name + '_sample.png')


if __name__ == '__main__':

    image_name = "/data/Capstone/test/test_data/test_2imgs/IMG_0020.png"
    config_path = 'configs/latent-diffusion/shanghai_gaus.yaml' # origin: crossattn, 3: concate
    
    ckpt_path = 'logs/' + '2023-05-26T16-05-56_shanghai_gaus' + '/checkpoints/epoch=000152.ckpt' # concate
    
    # ckpt_path = 'logs/2023-05-09T20-17-06_shanghai/checkpoints/last.ckpt' 

    inputimg = Image.open(image_name)
    # image = imageloader(inputimg, 256)
    dataset = ShanghaiTest(input_img=inputimg, size=512)
    ldm_cond_sample(config_path, ckpt_path, batch_size=1, out_name = image_name)

    # ldm_cond_sample(config_path, ckpt_path, 1, image)
