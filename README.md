

# AHGNN

Pytorch implementation of paper [A Novel Adaptive Hypergraph Neural Network for Enhancing Medical Image Segmentation](https://papers.miccai.org/miccai-2024/paper/2689_paper.pdf).

Our work has been accepted by MICCAI 2024.


![overview](https://github.com/11yxk/AHGNN/blob/main/overview.png)
# Data Preparation
We borrow the data process from [TransUnet](https://github.com/Beckschen/TransUNet).

# Training & Testing

### Training
```
python train.py --name AHGNN --base_lr 0.002 --batch_size 8 --max_epochs 600
```
### Testing

```
python test_loop_ds.py
```

### Pretrained Models

- Pretrained Models are available at https://drive.google.com/file/d/1oRRGuq_eDWEjauZZmowLu4m_57W3xqCc/view?usp=sharing .


## Acknowledgements

This repo is based on [TransUnet](https://github.com/Beckschen/TransUNet).

Thanks original authors for their work!
