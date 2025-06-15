import os
import torch
from models.base import CLIPBase


class CLIPLinear(CLIPBase):
    def __init__(
        self,
        device="cuda",
        base_model="openai/clip-vit-base-patch32",
        num_classes=1,
        classifier_path=None,
        use_processor=False,
    ):
        super(CLIPLinear, self).__init__(
            device, base_model, num_classes, classifier_path, use_processor
        )
        self.clip.eval()

    def save(self, path):
        torch.save(self.classifier.state_dict(), os.path.join(path, "classifier.pth"))
