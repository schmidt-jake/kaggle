# Notes

## DICOM format edge cases

1. Pixels are in 8-bit range, but window is a different bit depth, VOI LUT fn is none/linear. [Example](mammography/data/raw/train_images/32553/83977.dcm). Applying windowing will change the dtype to `float64` *and may shift the pixels by 1*.
1. Pixel range exceeds `BitsStored`. Bad data. Limited to patient_id 27770. [Example](mammography/data/raw/train_images/27770/176859678.dcm).
1. Low pixel range. Only affects image 1942326353. [Example](mammography/data/raw/train_images/822/1942326353.dcm)

## Visual edge cases

1. Low contrast
    1. 2057295788
    1. 610638958
    1. 2027498278
    1. 858907922 — implant
1. Nipple markers
    1. 2030427504
1. Circles
    1. 2049957533
1. Other artifacts
    1. 975386553

## Kaggle compute resources

- No accelerator
  - CPU cores: 4
  - Total memory: 32880876 kB
  - PyTorch version: 1.11.0+cpu
- 2 x T4
  - CPU cores: 2
  - Total memory: 16390860 kB
  - CUDA version: 11.4
  - per-GPU RAM: 15109MiB
  - PyTorch version: 1.11.0
- 1 x P100
  - CPU cores: 2
  - Total memory: 16390868 kB
  - CUDA version: 11.4
  - GPU RAM: 16280MiB
  - PyTorch version: 1.11.0
