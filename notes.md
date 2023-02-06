# Notes

## DICOM format edge cases

1. Pixels are in 8-bit range, but window is a different bit depth, VOI LUT fn is none/linear. [Example](mammography/data/raw/train_images/32553/83977.dcm). Applying windowing will change the dtype to `float64` *and may shift the pixels by 1*.
1. Pixel range exceeds `BitsStored`. Bad data. Limited to patient_id 27770. [Example](mammography/data/raw/train_images/27770/176859678.dcm).
1. Low pixel range. Only affects image 1942326353. [Example](mammography/data/raw/train_images/822/1942326353.dcm)

## Visual edge cases

1. Low contrast
    1. 858907922 — implant
1. Nipple markers
    1. 2030427504
1. Circles
    1. 2049957533
1. Overcropped
    1. 1181948538
    1. 1428845821
    1. 2121614268
1. Artificial border
    1. 47305213
    1. 42887848
    1. 33429299
    1. 32630261
    1. 2124758121
    1. 29003881
    1. 26934018
1. Other artifacts
    1. 1942326353 — all zeros
    1. 495314563 — high background value
    1. 1760744211 — two bright horizontal stripes

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

## Profiling

### Image preprocessing

Pydicom is much slower than dicomsdl + numexpr

Dicomsdl results for processing 100 files:

| OpenCV threads | Numexpr threads | Processes | context | wall time |
|----------------|-----------------|-----------|---------|-----------|
|       0        |        1        |     8     |   fork  |    33.5   |
|       0        |        1        |     8     |  spawn  |    38.3   |
|       0        |        2        |     8     |   fork  |    34.0   |
|       0        |        8        |     8     |   fork  |    34.6   |
|       0        |        8        |     8     |  spawn  |    35.9   |
|       0        |        1        |     4     |   fork  |    35.2   |
|       0        |        1        |     2     |   fork  |    53.3   |
|       0        |        2        |     4     |   fork  |    33.9   |

## Modeling ideas

### Feb 5

1. Try [`torchvision.ops.sigmoid_focal_loss`](https://pytorch.org/vision/main/generated/torchvision.ops.sigmoid_focal_loss.html#torchvision.ops.sigmoid_focal_loss)
1. Don't upsample the positive class quite so hard, make the upsampling ratio a hyperparameter
1. Use a larger initial crop size, e.g. 4096
1. Use a larger final resized crop, e.g. 1024

### Feb 6

These three ideas in combo seem to yield promising results:

1. Only train on CC and MLO views, which are by far the most standard.
    1. In the case that a breast has multiple instances of each view, randomly sample one.
1. Use a separate feature extractor for each of the views
1. Combine the predictions for each view (probably using `max`)

Next up:

1. Remove implants from training. This would reduce dataset size by 3% and require a different method for predicting for implants.
1. Add age as a predictor.
1. Predict density as an auxiliary task. This could help the model distinguish cancer signal from density signal.
    1. Furthermore, we could feed the density value (or density prediction, when not training) into the cancer predictor as side input.
1. If available, use more/all additional instances of each CC/MLO view
    1. First, explore what these additional views look like
1. Train on raw data (no VOI LUT)
