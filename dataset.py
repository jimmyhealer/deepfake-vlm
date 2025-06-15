import os
from PIL import Image
from torch.utils.data import Dataset


class DeepfakeVideoDataset(Dataset):
    """
    Dataset for deepfake video detection based on the FaceForensics++ dataset structure.
    It loads frames from videos and performs a video-level split for train, validation, and test sets.

    Splits are as follows:
    - Train: 80% of Real_youtube and 90% of FaceSwap.
    - Validation: 10% of Real_youtube and 10% of FaceSwap.
    - Test: 10% of Real_youtube and 100% of NeuralTextures.
    - Test_face2face: 100% of Face2Face.

    Labels:
    - 0: Fake
    - 1: Real

    Args:
        root_dir (str): Root directory of the dataset.
        split (str): The dataset split, one of 'train', 'val', 'test', or 'test_face2face'.
        transform (callable, optional): A function/transform to be applied to each frame.
        max_length (int, optional): Maximum number of frames to load.
    """

    def __init__(self, root_dir, split="train", transform=None, max_length=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.max_length = max_length
        self.frames = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        """
        Performs a video-level split of the dataset and loads the frames
        for the selected split.
        """
        # Define paths for different data categories
        real_youtube_path = os.path.join(self.root_dir, "Real_youtube")
        faceswap_path = os.path.join(self.root_dir, "FaceSwap")
        neuraltextures_path = os.path.join(self.root_dir, "NeuralTextures")
        face2face_path = os.path.join(self.root_dir, "manipulated_face2face_output")

        # Get video IDs for each category
        real_videos = (
            sorted(
                [
                    d
                    for d in os.listdir(real_youtube_path)
                    if os.path.isdir(os.path.join(real_youtube_path, d))
                ]
            )
            if os.path.isdir(real_youtube_path)
            else []
        )
        faceswap_videos = (
            sorted(
                [
                    d
                    for d in os.listdir(faceswap_path)
                    if os.path.isdir(os.path.join(faceswap_path, d))
                ]
            )
            if os.path.isdir(faceswap_path)
            else []
        )
        neural_videos = (
            sorted(
                [
                    d
                    for d in os.listdir(neuraltextures_path)
                    if os.path.isdir(os.path.join(neuraltextures_path, d))
                ]
            )
            if os.path.isdir(neuraltextures_path)
            else []
        )
        # Extract unique video IDs from face2face filenames
        face2face_videos = []
        if os.path.isdir(face2face_path):
            files = [f for f in os.listdir(face2face_path) if f.endswith(".jpg")]
            video_ids = set()
            for filename in files:
                parts = filename.split("_")
                if len(parts) >= 4 and parts[2] == "frame":
                    video_id = f"{parts[0]}_{parts[1]}"
                    video_ids.add(video_id)
            face2face_videos = sorted(list(video_ids))

        # Split Real_youtube videos
        num_real = len(real_videos)
        train_end_real = int(0.8 * num_real)
        val_end_real = int(0.9 * num_real)
        real_train = real_videos[:train_end_real]
        real_val = real_videos[train_end_real:val_end_real]
        real_test = real_videos[val_end_real:]

        # Split FaceSwap videos
        num_faceswap = len(faceswap_videos)
        train_end_faceswap = int(0.9 * num_faceswap)
        faceswap_train = faceswap_videos[:train_end_faceswap]
        faceswap_val = faceswap_videos[train_end_faceswap:]

        # Select videos based on the specified split
        video_list = []
        if self.split == "train":
            video_list.extend(
                [(os.path.join(real_youtube_path, vid), 0, vid) for vid in real_train]
            )
            video_list.extend(
                [(os.path.join(faceswap_path, vid), 1, vid) for vid in faceswap_train]
            )
        elif self.split == "val":
            video_list.extend(
                [(os.path.join(real_youtube_path, vid), 0, vid) for vid in real_val]
            )
            video_list.extend(
                [(os.path.join(faceswap_path, vid), 1, vid) for vid in faceswap_val]
            )
        elif self.split == "test":
            video_list.extend(
                [(os.path.join(real_youtube_path, vid), 0, vid) for vid in real_test]
            )
            video_list.extend(
                [
                    (os.path.join(neuraltextures_path, vid), 1, vid)
                    for vid in neural_videos
                ]
            )
        elif self.split == "test_face2face":
            video_list.extend(
                [(os.path.join(real_youtube_path, vid), 0, vid) for vid in real_test]
            )
            video_list.extend([(face2face_path, 1, vid) for vid in face2face_videos])
        else:
            raise ValueError(
                f"Invalid split '{self.split}'. Choose 'train', 'val', 'test', or 'test_face2face'."
            )

        # Load all frames from the selected videos
        self._load_frames(video_list)

    def _load_frames(self, video_list):
        """
        Scans the video directories and populates the frame list.
        """
        # Pre-process face2face frames if this is a face2face split
        face2face_path_for_pre_scan = None
        face2face_frames_by_video_id = {}

        if self.split == "test_face2face":
            # Find the common face2face directory path from the video_list
            for path, _, _ in video_list:
                if "face2face" in path.lower():
                    face2face_path_for_pre_scan = path
                    break

            if face2face_path_for_pre_scan and os.path.isdir(
                face2face_path_for_pre_scan
            ):
                # Scan the face2face directory once to build the map
                for filename in sorted(os.listdir(face2face_path_for_pre_scan)):
                    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                        parts = filename.split("_")
                        if len(parts) >= 4 and parts[2] == "frame":
                            file_video_id = f"{parts[0]}_{parts[1]}"
                            if file_video_id not in face2face_frames_by_video_id:
                                face2face_frames_by_video_id[file_video_id] = []
                            face2face_frames_by_video_id[file_video_id].append(
                                os.path.join(face2face_path_for_pre_scan, filename)
                            )

        for video_path, label, video_id in video_list:
            if not os.path.isdir(video_path):
                print(f"Warning: Directory not found, skipping: {video_path}")
                continue

            # Special handling for face2face dataset
            if self.split == "test_face2face" and "face2face" in video_path.lower():
                if video_id in face2face_frames_by_video_id:
                    for frame_path in face2face_frames_by_video_id[video_id]:
                        if (
                            self.max_length is not None
                            and self.max_length > 0
                            and len(self.frames) >= self.max_length
                        ):
                            return
                        self.frames.append((frame_path, label, video_id))
            else:
                for filename in sorted(os.listdir(video_path)):
                    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                        if (
                            self.max_length is not None
                            and self.max_length > 0
                            and len(self.frames) >= self.max_length
                        ):
                            return
                        self.frames.append(
                            (os.path.join(video_path, filename), label, video_id)
                        )

    def __len__(self):
        """Returns the number of frames in the dataset."""
        return len(self.frames)

    def __getitem__(self, idx):
        """
        Returns a single frame, its label, and its video ID.

        Args:
            idx (int): Index of the frame.

        Returns:
            dict: A dictionary containing the image, label, and video_id.
        """
        frame_path, label, video_id = self.frames[idx]

        try:
            image = Image.open(frame_path).convert("RGB")
            if self.transform:
                image = self.transform(image)

            return {"image": image, "label": label, "video_id": video_id}
        except IOError as e:
            print(f"Warning: Error opening image file {frame_path}: {e}. Skipping.")
            return {"image": None, "label": None, "video_id": None}


if __name__ == "__main__":
    # This block is for demonstration and verification purposes.
    # TODO: Adjust this path to your actual dataset directory.
    root_dir = "datasets"

    print("Initializing datasets...")
    # Check if the root directory exists before proceeding
    if not os.path.isdir(root_dir):
        print(f"Error: Dataset root directory not found at '{root_dir}'")
        print("Please download the dataset or adjust the 'root_dir' variable.")
    else:
        # Create dataset instances for each split
        train_dataset = DeepfakeVideoDataset(root_dir=root_dir, split="train")
        val_dataset = DeepfakeVideoDataset(root_dir=root_dir, split="val")
        test_dataset = DeepfakeVideoDataset(root_dir=root_dir, split="test")

        print(f"Train dataset size: {len(train_dataset)} frames")
        print(f"Validation dataset size: {len(val_dataset)} frames")
        print(f"Test dataset size: {len(test_dataset)} frames")

        # --- Verification for Data Leaks ---
        print("\nVerifying data splits to prevent leaks...")

        # Get video IDs from each dataset split
        train_video_ids = {item[2] for item in train_dataset.frames}
        val_video_ids = {item[2] for item in val_dataset.frames}
        test_video_ids = {item[2] for item in test_dataset.frames}

        # Get all video IDs from the NeuralTextures directory
        neuraltextures_path = os.path.join(root_dir, "NeuralTextures")
        neural_video_ids = (
            set(
                [
                    d
                    for d in os.listdir(neuraltextures_path)
                    if os.path.isdir(os.path.join(neuraltextures_path, d))
                ]
            )
            if os.path.isdir(neuraltextures_path)
            else set()
        )

        # Check that videos from the NeuralTextures *directory* are only in the test set
        train_neural_intersection = {
            item[2] for item in train_dataset.frames if "NeuralTextures" in item[0]
        }
        val_neural_intersection = {
            item[2] for item in val_dataset.frames if "NeuralTextures" in item[0]
        }

        if not train_neural_intersection and not val_neural_intersection:
            print(
                "PASSED: No frames from the NeuralTextures directory are in the train or validation sets."
            )
        else:
            print(
                "FAILED: Frames from the NeuralTextures directory leaked into train/val sets."
            )
            if train_neural_intersection:
                print(
                    f"   Leaked into train set (video IDs): {train_neural_intersection}"
                )
            if val_neural_intersection:
                print(
                    f"   Leaked into validation set (video IDs): {val_neural_intersection}"
                )

        # Verify that the splits for Real and FaceSwap videos are disjoint
        real_youtube_path = os.path.join(root_dir, "Real_youtube")
        faceswap_path = os.path.join(root_dir, "FaceSwap")

        train_real_fs_ids = {
            item[2]
            for item in train_dataset.frames
            if real_youtube_path in item[0] or faceswap_path in item[0]
        }
        val_real_fs_ids = {
            item[2]
            for item in val_dataset.frames
            if real_youtube_path in item[0] or faceswap_path in item[0]
        }
        test_real_fs_ids = {
            item[2]
            for item in test_dataset.frames
            if real_youtube_path in item[0] or faceswap_path in item[0]
        }

        train_val_leak = train_real_fs_ids.intersection(val_real_fs_ids)
        train_test_leak = train_real_fs_ids.intersection(test_real_fs_ids)
        val_test_leak = val_real_fs_ids.intersection(test_real_fs_ids)

        if not train_val_leak and not train_test_leak and not val_test_leak:
            print(
                "PASSED: Train, validation, and test splits are disjoint for Real and FaceSwap videos."
            )
        else:
            print(
                "FAILED: Overlap detected between splits for Real and FaceSwap videos."
            )
            if train_val_leak:
                print(f"   Overlap between train and val: {train_val_leak}")
            if train_test_leak:
                print(f"   Overlap between train and test: {train_test_leak}")
            if val_test_leak:
                print(f"   Overlap between val and test: {val_test_leak}")
