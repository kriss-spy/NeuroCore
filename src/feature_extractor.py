import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from tqdm import tqdm
from src.data_utils import load_aloha_dataset


def get_feature_extractor(device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Initializes a pre-trained ResNet-18 model, strips the final FC layer,
    and returns it along with the appropriate ImageNet transforms.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    # Replace the final fully connected layer with an Identity module to get 512-D features
    model.fc = nn.Identity()  # type: ignore
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    # Standard ImageNet normalization and sizing
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return model, preprocess, device


def extract_and_cache_features(save_path="results/features_resnet18.pt"):
    """
    Extracts ResNet-18 features for all frames across all episodes and caches them to disk.
    Returns the dictionary mapping (episode_index, frame_index) -> 512-D feature tensor.
    """
    if os.path.exists(save_path):
        print(f"Features already cached at {save_path}. Loading from disk...")
        return torch.load(save_path, weights_only=True)

    model, preprocess, device = get_feature_extractor()
    dataset = load_aloha_dataset()

    # Identify the correct image key (e.g., 'observation.images.top' or similar)
    image_keys = [k for k in dataset[0].keys() if "image" in k]
    if not image_keys:
        raise ValueError("No image features found in the dataset.")
    image_key = image_keys[0]  # We use the first available camera view

    print(f"Extracting features from '{image_key}' on {device}...")

    features_dict = {}

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Extracting features"):
            item = dataset[i]
            ep_idx = item["episode_index"]
            frame_idx = item["frame_index"]

            img = item[image_key]
            # Convert PIL image using transforms
            img_tensor = preprocess(img).unsqueeze(0).to(device)

            # Extract 512-D feature vector
            feat = model(img_tensor).squeeze(0).cpu()

            features_dict[(int(ep_idx), int(frame_idx))] = feat

    # Ensure directory exists and cache the results
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(features_dict, save_path)
    print(f"Successfully saved {len(features_dict)} features to {save_path}")

    return features_dict
