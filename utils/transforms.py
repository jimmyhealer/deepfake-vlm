from torchvision import transforms
import torch


def get_transforms(processor, data_aug=False):
    """
    Get the transform for the CLIP model.
    Args:
        processor: CLIPProcessor
        data_aug: bool, whether to use data augmentation

    Returns:
        transform: transforms.Compose
        data_aug_transform: transforms.Compose
    """
    transform = transforms.Compose(
        [
            transforms.Resize(processor.image_processor.size["shortest_edge"]),
            transforms.CenterCrop(processor.image_processor.crop_size["height"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=processor.image_processor.image_mean,
                std=processor.image_processor.image_std,
            ),
        ]
    )

    data_aug_transform = transforms.Compose(
        [
            transforms.Resize(processor.image_processor.size["shortest_edge"]),
            transforms.RandomResizedCrop(
                processor.image_processor.crop_size["height"], scale=(0.9, 1.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=processor.image_processor.image_mean,
                std=processor.image_processor.image_std,
            ),
            transforms.Lambda(lambda x: x + 0.005 * torch.randn_like(x)),
        ]
    )

    if data_aug:
        return transform, data_aug_transform
    else:
        return transform, transform
