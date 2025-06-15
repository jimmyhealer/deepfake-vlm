import torch
import os
from peft import get_peft_model, LoraConfig, PeftModel

from models.base import CLIPBase


class CLIPLora(CLIPBase):
    """
    CLIP model with a LoRA adapter and a linear classification head.
    """

    def __init__(
        self,
        device="cuda",
        base_model="openai/clip-vit-base-patch32",
        adapter_path=None,
        classifier_path=None,
        num_classes=1,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        use_processor=False,
    ):
        super(CLIPLora, self).__init__(
            device, base_model, num_classes, classifier_path, use_processor
        )

        if adapter_path is not None:
            self.clip.vision_model = PeftModel.from_pretrained(
                self.clip.vision_model, adapter_path
            )
        else:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            self.clip.vision_model = get_peft_model(self.clip.vision_model, lora_config)

    def save(self, path):
        self.clip.vision_model.save_pretrained(os.path.join(path, "lora_adapters"))
        torch.save(self.classifier.state_dict(), os.path.join(path, "classifier.pth"))