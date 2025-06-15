import torch
import torch.nn as nn
from transformers import AutoProcessor, CLIPModel

from utils.transforms import get_transforms


class CLIPBase(nn.Module):
    def __init__(
        self,
        device="cuda",
        base_model="openai/clip-vit-base-patch32",
        num_classes=1,
        classifier_path=None,
        use_processor=False,
    ):
        super(CLIPBase, self).__init__()
        self.device = device
        self.base_model = base_model
        self.use_processor = use_processor
        self.processor = AutoProcessor.from_pretrained(base_model, use_fast=True)
        self.clip = CLIPModel.from_pretrained(base_model).to(device)

        for param in self.clip.parameters():
            param.requires_grad = False

        embedding_dim = self.clip.config.projection_dim
        self.classifier = nn.Linear(embedding_dim, num_classes).to(device)

        if classifier_path is not None:
            self.classifier.load_state_dict(
                torch.load(classifier_path, map_location=device)
            )

        if not use_processor:
            self.eval_transform, self.train_transform = get_transforms(self.processor)
        else:
            self.eval_transform = None
            self.train_transform = None

    def forward(self, pixel_values):
        """
        Forward pass for the CLIPLora model.
        """
        outputs = self.clip.get_image_features(pixel_values=pixel_values)
        logits = self.classifier(outputs)
        return logits.squeeze(-1)

    def preprocess(self, images):
        if self.use_processor:
            return self.processor.image_processor(images, return_tensors="pt")[
                "pixel_values"
            ].to(self.device)

        return images.to(self.device)

    def print_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        percent = trainable_params / total_params * 100

        return total_params, trainable_params, percent
