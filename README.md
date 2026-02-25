# ViPSAM : Visaul Prompting Medical Image Segmentation Using Segment Anything Model

The official source code of "ViPSAM : Visaul Prompting Medical Image Segmentation Using Segment Anything Model".

# Implementation

A Pytorch implementation of deep-learning-based model.

- Requirements
  - OS : Ubuntu / Windows
  - Python 3.10.18
  - PyTorch 2.5.1

# Dataset

- In our experiment, we used a proton therapy planning dataset consisting of paired NCCT and contrast-enhanced MRI scans collected at a tertiary medical center.
- The dataset comprises 73 cases, each including a mid-respiratory phase NCCT (T = 50%), a T1-weighted fat-suppressed contrast-enhanced MRI at the same respiratory phase, and expert-annotated liver and lesion masks defined on the NCCT.
- MRI scans were rigidly registered to the NCCT, and all scans were resampled to a voxel spacing of 0.6597 × 0.6597 × 5mm.

# Evaluation

- test.py is the implementation code if inference for liver lesion segmentation
