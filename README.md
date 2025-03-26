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
conda env create -f environment.yaml
conda activate ldm
```
* 코드 구조
<br>![pic1](assets/architecture.jpg)


# Run

```
python latent-diffusion/main.py --base configs/latent-diffusion/shanghai_amh.yaml -t --gpus 0, 
```
