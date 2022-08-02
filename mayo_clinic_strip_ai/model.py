"""
Defines the neural network architecture
"""
from typing import Optional

import torch
from torchvision import models


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


class FeatureExtractor(torch.nn.Module):
    def __init__(self, backbone_fn: str, weights: Optional[str] = None):
        """
        Extracts a feature vector from a normalized RGB image, using a backbone architecture.

        Parameters
        ----------
        backbone_fn : str
            A valid callable attribute on `torchvision.models`
        weights : Optional[str]
            A string to be passed to `torchvision.models.get_weights`, default None
            Example: "ResNet50_Weights.IMAGENET1K_V1"
        """
        super().__init__()
        if weights is not None:
            weights = models.get_weight(weights)
        backbone: torch.nn.Module = getattr(models, backbone_fn)(weights=weights)
        self.features: torch.nn.Sequential = backbone.features  # type: ignore[assignment]
        self.features.append(torch.nn.ReLU(inplace=True))
        self.features.append(torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.features.append(torch.nn.Flatten(start_dim=1))

    # @torch.jit.script_method
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        features = self.features(img)
        return features


class Classifier(torch.nn.Module):
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
        super().__init__()
        self.logit = torch.nn.Linear(in_features=in_features, out_features=1)
        self.logit.bias = torch.nn.Parameter(
            torch.tensor(initial_logit_bias, requires_grad=True),
            requires_grad=True,
        )

    # @torch.jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logit(x)


class Model(torch.nn.Module):
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
        super().__init__()
        self.normalizer = normalizer
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.activation = torch.nn.ReLU(inplace=True)

    # @torch.jit.script_method
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.normalizer(img)
        x = self.feature_extractor(x)
        x = self.activation(x)
        logit: torch.Tensor = self.classifier(x)
        return logit


class Loss(torch.nn.Module):
    def __init__(self, pos_weight: float):
        super().__init__()
        self.bce_logit_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    # @torch.jit.script_method
    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        label = label.unsqueeze(dim=1)
        label = label.to(dtype=logit.dtype)
        return self.bce_logit_loss(input=logit, target=label)
