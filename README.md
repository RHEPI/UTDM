# UTDM: A Universal Transformer-based  Diffusion Model for Multi-weather Degraded Images Restoration

This is the code repository of the following paper.

"UTDM: A Universal Transformer-based  Diffusion Model for Multi-weather Degraded Images Restoration"\
<em>Yongbo Yu, Weidong Li, Linyan Bai, Jinlong Duan, Xuehai Zhang</em>

> **Abstract**: Restoring multi-weather degraded images is significant for subsequent high-level computer vision tasks. However, most existing image restoration algorithms only target single-weather degraded images, and there are few general models for multi-weather degraded image restoration. In this paper, we propose a diffusion model for multi-weather degraded image restoration, namely A Universal Transformer-based  Diffusion Model (UTDM) for Multi-weather Degraded Images Restoration, by combining the denoising diffusion probability model (DDPM) and Vision Transformer (ViT). First, UTDM uses weather-degraded images as conditions to guide the diffusion model to generate clean background images through reverse sampling. Secondly, we propose a Cascaded Fusion Noise Estimation Transformer (CFNET) based on ViT, which utilizes degraded and noisy images for noise estimation. By introducing Cascaded Contextual Fusion Attention (CCFA) in a cascaded manner to compute contextual fusion attention mechanisms for different heads, CFNET explores the commonalities and characteristics of multi-weather degraded images, fully capturing global and local feature information to improve the model's generalization ability on various weather-degraded images. UTDM outperformed the existing algorithm by 0.14~4.55dB on the Raindrop-A test set, and improved by 0.99dB and 1.24dB compared with Transweather on the Snow100K-L and Test1 test sets. Experimental results show that our method outperforms general and specific restoration task algorithms on synthetic and real-world degraded image datasets.

## Using the code:

The code is stable while using Python 3.8.5, CUDA >=11.7

Clone this repository:

```yaml
git clone https://github.com/RHEPI/UTDM.git
cd UTDM
```

To install all the dependencies using conda:

```yaml
conda env create -f environment.yml
conda activate UTDM
```

If you prefer pip, install following versions:

```yaml
timm==0.9.12
torch==2.0.1+cu117
torchvision==0.15.2
opencv-python==4.8.1.78
pillow==9.4.0
yaml==0.2.5
tqdm==4.65.0
```

## Datasets

To realize a general model for multi-weather degraded image restoration, the model in this paper is trained on TransWeather's [AllWeather](https://github.com/jeya-maria-jose/TransWeather) dataset, which consists of subsets of the following three benchmark datasets. 

1. The image de-snowing dataset [Snow100K](https://sites.google.com/view/yunfuliu/desnownet) from DesnowNet image snow removal dataset Snow100K.
2. Raindrop removal dataset [RainDrop](https://github.com/rui1996/DeRaindrop).
3. Image de-raining and de-fogging dataset [Outdoor-Rain](https://github.com/liruoteng/HeavyRainRemoval).

### Real-World Datasets

After training with TransWeather's AllWeather dataset, this paper uses the real-world raindrop dataset [RainDS](https://drive.google.com/file/d/12yN6avKi4Tkrnqa3sMUmyyf4FET9npOT/view) dataset as well as the [Snow100K Realistic snowy images](https://www.google.com/url?q=https%3A%2F%2Fdesnownet.s3.amazonaws.com%2Frealistic_image%2Frealistic.tar.gz&sa=D&sntz=1&usg=AOvVaw3SrhOt805ebXPoHQ6ruFqi) to validate the model's generalization performance in the real world.

## Training and Evaluation

### Train

```
python train.py
```

### Test

```
python test.py
```

For model training parameter settings, you can find `allweather.yml` as well as `allweather128.yml` in configs.

## Model Structure

Architecture of the MDIR-UCDM Mode:
<img src="imgs/fig1.jpg">

### Image Restoration

Visual comparison of different algorithms on the **Raindrop-A** dataset:
<img src="imgs/fig2.png">

Visual comparison of different algorithms on the **Outdoor-Rain** dataset:
<img src="imgs/fig3.png">

Visual comparison of different algorithms on the **Snow100K** dataset:
<img src="imgs/fig4.png">

## Acknowledgments

The authors express gratitude to the Key Scientific Research Projects of Colleges and Universities in Henan Province (NO.23A170013). We would like to express our appreciation to the National Supercomputing Center in Zhengzhou for providing the necessary computaional platform required for our experiment.

Parts of this code repository is based on the following works:
* https://github.com/IGITUGraz/WeatherDiffusion
* https://github.com/facebookresearch/DiT
* https://github.com/JDAI-CV/CoTNet

## Reference

If you find this work useful for your research, please cite our paper.
