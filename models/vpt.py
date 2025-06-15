from models.base import CLIPBase
from torch import nn
import torch
import os


class VisualPrompt(nn.Module):
    """A small, trainable patch to be applied to images."""

    def __init__(self, prompt_size, image_channels=3):
        super().__init__()
        self.prompt_size = prompt_size
        self.prompt = nn.Parameter(
            torch.randn(1, image_channels, prompt_size, prompt_size)
        )

    def forward(self, x):
        x[:, :, : self.prompt_size, : self.prompt_size] = self.prompt
        return x


class CLIPVPT(CLIPBase):
    def __init__(
        self,
        device="cuda",
        base_model="openai/clip-vit-base-patch32",
        prompt_path=None,
        classifier_path=None,
        num_classes=1,
        prompt_size=30,
        use_processor=False,
    ):
        super(CLIPVPT, self).__init__(
            device=device,
            base_model=base_model,
            num_classes=num_classes,
            use_processor=use_processor,
            classifier_path=None,  # Not loading via base
        )

        embedding_dim = self.clip.vision_model.config.hidden_size
        self.classifier = nn.Linear(embedding_dim, num_classes).to(device)

        if classifier_path:
            self.classifier.load_state_dict(
                torch.load(classifier_path, map_location=device)
            )

        self.visual_prompt = VisualPrompt(prompt_size).to(device)
        if prompt_path:
            self.visual_prompt.load_state_dict(
                torch.load(prompt_path, map_location=device)
            )

        # Freeze the entire CLIP model
        for param in self.clip.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        pixel_values = self.visual_prompt(pixel_values)
        outputs = self.clip.vision_model(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        if self.classifier.out_features == 1:
            return logits.squeeze(-1)
        return logits

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(
            self.visual_prompt.state_dict(), os.path.join(path, "visual_prompt.pth")
        )
        torch.save(self.classifier.state_dict(), os.path.join(path, "classifier.pth"))
