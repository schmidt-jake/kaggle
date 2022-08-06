"""
Defines the neural network architecture
"""


import torch


class Normalizer(torch.nn.Module):
    # @torch.jit.script_method
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Normalizes an input image.

        Parameters
        ----------
        img : torch.Tensor
            The input image minibatch, NHWC.

        Returns
        -------
        torch.Tensor
            The normalized image batch, NCHW.
        """
        # img = img.clone()
        # _min = img.amin(dim=(1, 2), keepdim=True)
        # _max = img.amax(dim=(1, 2), keepdim=True)
        # img = (img - _min) / (_max - _min)
        img = img / 255
        return img


class FeatureExtractor(torch.nn.Sequential):
    def __init__(self, backbone: torch.nn.Module):
        """
        Extracts a feature vector from a normalized RGB image, using a backbone architecture.

        Parameters
        ----------
        backbone: a model from `torchvision.models`.
        """
        super().__init__(
            backbone.features,  # type: ignore[arg-type]
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.Flatten(start_dim=1),
        )


class Classifier(torch.nn.Sequential):
    def __init__(self, initial_logit_bias: float, in_features: int):
        """
        A linear binary classifier.

        Parameters
        ----------
        initial_logit_bias : float
            An initial bias value for the classifier. Typically used to bias the classifier output
            at the start of training in the case of class imbalance.
        in_features : int
            The dimension of the input to the classifier.
        """
        logit = torch.nn.Linear(in_features=in_features, out_features=1)
        logit.bias = torch.nn.Parameter(
            torch.tensor(initial_logit_bias, requires_grad=True),
            requires_grad=True,
        )
        super().__init__(torch.nn.BatchNorm1d(in_features), logit)


class Model(torch.nn.Sequential):
    def __init__(self, normalizer: Normalizer, feature_extractor: FeatureExtractor, classifier: Classifier):
        """
        A binary classifier over RGB images.

        Parameters
        ----------
        normalizer : Normalizer
            A module that preprocesses the images
        feature_extractor : FeatureExtractor
            A module that extracts feature vectors from normalized images
        classifier : Classifier
            A binary classifier.
        """
        super().__init__(normalizer, feature_extractor, classifier)


class Loss(torch.nn.Module):
    def __init__(self, pos_weight: float):
        super().__init__()
        self.bce_logit_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    # @torch.jit.script_method
    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        label = label.unsqueeze(dim=1)
        label = label.to(dtype=logit.dtype)
        return self.bce_logit_loss(input=logit, target=label)
