# Notes

RSNA Breast Cancer competition source data GCS path: `gs://kds-93187e560c991ac18dcbcb5b4e86d922f1b4d72992dc07be698e602b`

## DICOM format edge cases

1. Pixels are in 8-bit range, but window is a different bit depth, VOI LUT fn is none/linear. [Example](mammography/data/raw/train_images/32553/83977.dcm). Applying windowing will change the dtype to `float64` *and may shift the pixels by 1*.
1. Pixel range exceeds `BitsStored`. Bad data. Limited to patient_id 27770. [Example](mammography/data/raw/train_images/27770/176859678.dcm).
1. Low pixel range. Only affects image 1942326353. [Example](mammography/data/raw/train_images/822/1942326353.dcm)

## Visual edge cases

197998560 — just weird, probably unusable
1634189725 — ^
802890376 — unusable without manual cropping
936880795 — unusable without manual cropping
1074322983 — unusable without manual cropping
1249735234 — unusable without manual cropping
1839672250 — weird artifact, needs manual cropping

1189630231 — horizontal lines
764545189 — horizontal lines

682364838 — weird shadow effect?
2011677381 — ^

743466894 — weird vertical line

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
    1. 2051385093
    1. 1521956184
    1. 1223335281
    1. 1181635673
    1. 871227326
    1. 381072284
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
    1. 764545189 — two bright horizontal stripes
    1. 597771506 — pacemaker
    1. 355971526 — pacemaker
    1. 189841090 — pacemaker
    1. 213706302 — pacemaker
    1. 321321852 — pacemaker
    1. 330666379 — pacemaker
    1. 355971526 — pacemaker
    1. 412697751 — pacemaker
    1. 613969302 — pacemaker
    1. 744730557 — pacemaker
    1. 748668947 — idk
    1. 800715529 — pacemaker
1. Possible false negative implant
    1. see [`implants.json`](implants.json)

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

## Other Notes

- Site 2 has no `density` data.
- `age` can be NaN.
- `machine_id`s 190, 1070, and 216 are missing `PixelPaddingValue` for all images. All other `machine_id`s have `PixelPaddingValue` for all images.
