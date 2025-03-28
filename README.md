# Capstone Design in ICT, Spring 2023
## git 사용
```
git checkout <브랜치>
git pull origin main
# 파일 수정
git commit -am "<코맨트>"
git push
# pull request
```

# Environment
```
conda env create -f latent-diffusion/environment.yaml
conda activate ldm
```
* 코드 구조
<br>![pic1](assets/architecture.jpg) 
[High-Resolution Image Synthesis with Latent Diffusion Models (CVPR 2022)](https://arxiv.org/abs/2112.10752)를 베이스라인으로 사용


# 데이터 구성 및 전처리
``density_shanghai.py`` 를 통해 원본 RGB 이미지에 Gaussian kernel을 적용하여 density map을 생성

Conditioning (조건 입력):
원본 RGB 이미지를 사용하며, 별도의 모델 없이 단순히 rescale 처리를 진행함

데이터셋으로는 ``Shanghaitech``를 사용하였음
``Shanghaitech`` 데이터셋에는 330165명의 머리에 해당하는 위치데이터가 같이 포함되어 있으며 Crowd Counting에 대표적으로 사용되는 표준 벤치마크 데이터셋임.
1198개의 이미지로 구성이 되어있으며 인터넷에서 수집된 Part-A, 상하이 번화가 거리에서 수집된 Part-B로 나누어져 있음.

- Part-A: 300개의 train 이미지와 182개의 test 이미지
- Part-B: 400개의 train 이미지와 316개의 test 이미지


데이터 경로:
```
ShanghaiTech_val/part_A/train_data/density/DENSITY_*.png   # Density map 경로
ShanghaiTech_val/part_A/train_data/img/IMG_*.png           # RGB 이미지 경로
```
데이터 로더:
``latent-diffusion/ldm/data/shanghai.py``
# Encoding
## Density Encoding (First Stage)
Pretrained VQ 모델 사용 https://ommer-lab.com/files/latent-diffusion/vq-f4.zip

특징:
  - 4x downsample (f = 4)
  - VQ embedding space: Z = 8192, latent dim d = 3

다운로드한 모델 체크포인트 ``model.ckpt``는 다음 경로에 저장
``latent-diffusion/models/first_stage_models/vq-f4/model.ckpt``

## RGB Conditioning Encoding
별도의 인코딩 모델 없이 RGB 이미지를 rescale하여 사용하거나, Pretrained 모델 (FrozenCLIPEmbedder 등)이 사용 가능함

# Train
```
python latent-diffusion/main.py --base configs/latent-diffusion/config.yaml -t --gpus 0, 
```
