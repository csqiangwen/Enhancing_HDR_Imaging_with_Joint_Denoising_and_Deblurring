# Enhancing HDR Imaging with Joint Denoising and Deblurring
[Paper](https://link.springer.com/content/pdf/10.1007/s11263-025-02537-w.pdf) | [Project Page](https://csqiangwen.github.io/projects/hdr-hidd/)

This is the official PyTorch implementation of ''Enhancing HDR Imaging with Joint Denoising and Deblurring''.

## Preparation
Python 3.8+  
Pytorch 1.4.0+

## Dataset ([link](https://hkustconnect-my.sharepoint.com/personal/qwenab_connect_ust_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fqwenab%5Fconnect%5Fust%5Fhk%2FDocuments%2FHDR%5FDenoising%5FDeblurring%2FHIDD%5FDataset&ga=1e))
We provide two training sets:
- "train": Training data at the original resolution.
- "train_patch": Training data is divided into patches to speed up the training process (default setting).

## Training
- This is an example of training.
```bash
$ bash train.sh
```

## Evaluation
- To test:
```
$ bash test.sh
```
- It will save HDR results with their corresponding GTs under "img_test_HDR_GT";
- During testing, we provide approximate metric scores (PSNR, SSIM) for reference.
- For quantitative evaluation, we use the PU21 metric to obtain the final scores (the same reported in the paper).

## Testing (your own data)
- To test:
```
$ bash test_custom.sh
```
- The custom dataset should follow the same structure as the HIDD test set, except it does not include the "GT" folder.

## Citation
If you find this repository useful for your research, please cite the following work.
```
@article{wen2025enhancing,
          title={Enhancing HDR Imaging with Joint Denoising and Deblurring},
          author={Wen, Qiang and Rao, Zhefan and Lei, Chenyang and Sun, Wenxiu and Yan, Qiong and Li, Jing and Lei, Fei and Chen, Qifeng},
          journal={International Journal of Computer Vision},
          pages={1--17},
          year={2025},
          publisher={Springer}
}

```
<p align='center'>
<img src='Logo/HKUST_VIL.png' width=500>
</p>
