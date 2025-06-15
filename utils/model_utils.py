import logging
import os

from models import CLIPLinear, CLIPLora, CLIPVPT


def select_model(args):
    """
    Selects and initializes a model based on command-line arguments.
    """
    if args.model == "clip_linear":
        classifier_path = None
        if "model_dir" in args and args.model_dir is not None:
            if not os.path.exists(args.model_dir):
                raise FileNotFoundError(
                    f"Model weights not found at {args.model_dir}"
                )
            classifier_path = os.path.join(args.model_dir, "classifier.pth")
            if not os.path.exists(classifier_path):
                raise FileNotFoundError(
                    f"Classifier weights not found at {classifier_path}"
                )
            logging.info(f"Loading classifier weights from {classifier_path}")

        return CLIPLinear(
            device=args.device,
            base_model=args.base_model,
            classifier_path=classifier_path,
            use_processor=args.use_processor,
        )
    elif args.model == "clip_lora":
        if "model_dir" in args and args.model_dir is not None:
            logging.info(
                f"Loading LoRA model with adapter weights from {args.model_dir}"
            )
            adapter_path = os.path.join(args.model_dir, "lora_adapters")
            classifier_path = os.path.join(args.model_dir, "classifier.pth")
            if not os.path.exists(adapter_path):
                raise FileNotFoundError(
                    f"Adapter weights not found at {adapter_path}"
                )
            if not os.path.exists(classifier_path):
                raise FileNotFoundError(
                    f"Classifier weights not found at {classifier_path}"
                )

            return CLIPLora(
                device=args.device,
                base_model=args.base_model,
                adapter_path=adapter_path,
                classifier_path=classifier_path,
                use_processor=args.use_processor,
            )

        logging.info(
            f"Loading LoRA model with r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, target_modules={args.lora_target_modules}"
        )

        return CLIPLora(
            device=args.device,
            base_model=args.base_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(","),
            use_processor=args.use_processor,
        )
    elif args.model == "clip_vpt":
        if "model_dir" in args and args.model_dir is not None:
            logging.info(f"Loading VPT model with weights from {args.model_dir}")
            prompt_path = os.path.join(args.model_dir, "visual_prompt.pth")
            classifier_path = os.path.join(args.model_dir, "classifier.pth")
            if not os.path.exists(prompt_path):
                raise FileNotFoundError(f"Prompt weights not found at {prompt_path}")
            if not os.path.exists(classifier_path):
                raise FileNotFoundError(
                    f"Classifier weights not found at {classifier_path}"
                )

            return CLIPVPT(
                device=args.device,
                base_model=args.base_model,
                prompt_size=args.prompt_size,
                prompt_path=prompt_path,
                classifier_path=classifier_path,
                use_processor=args.use_processor,
            )

        logging.info(f"Loading VPT model with prompt size={args.prompt_size}")

        return CLIPVPT(
            device=args.device,
            base_model=args.base_model,
            prompt_size=args.prompt_size,
            use_processor=args.use_processor,
        )
    else:
        raise ValueError(f"Model {args.model} not supported.")