"""
Language instruction embedding extraction using frozen CLIP text encoder.

Since the ALOHA Sim Transfer Cube dataset is a single-task dataset
(all episodes perform the same "transfer cube" manipulation), we assign
a unified task-level language instruction to all frames. This instruction
acts as the task goal, analogous to how the prefrontal cortex sends
downstream task goal signals to motor areas during goal-directed action.
"""

import os
from typing import Tuple

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

# The task-level language instruction for the ALOHA Transfer Cube task.
# In biological terms, this is the "task goal" held in working memory
# by the prefrontal cortex and broadcast to sensorimotor areas.
TASK_INSTRUCTION = (
    "Pick up the cube and transfer it to the target location."
)

# Path to cache the computed language embedding.
_LANG_EMBED_CACHE_PATH = "results/language_embedding.pt"

# Model identifier for the frozen CLIP text encoder.
_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


def extract_language_embedding(
    instruction: str = TASK_INSTRUCTION,
    cache_path: str = _LANG_EMBED_CACHE_PATH,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """
    Extract a fixed language embedding for the task instruction using a
    frozen CLIP text encoder.

    Brain-inspired design note:
    The prefrontal cortex (PFC) maintains task goals in working memory and
    broadcasts them to parietal and motor cortices during action execution.
    Here, the CLIP text encoder serves as a computational analog to the
    PFC's semantic goal representation, while the frozen weights ensure
    that the goal signal remains stable (non-trainable) during motor
    learning, consistent with biological observations that task goals are
    held constant during skill acquisition.

    Args:
        instruction: The natural language task description.
        cache_path: Where to save/load the cached embedding.
        device: Compute device.

    Returns:
        A 1-D tensor of shape (embed_dim,) representing the language
        instruction embedding.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached language embedding from {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    print(f"Extracting language embedding with {_CLIP_MODEL_NAME} ...")
    tokenizer = CLIPTokenizer.from_pretrained(_CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(_CLIP_MODEL_NAME).to(device)
    model.eval()

    # Freeze all parameters — the text encoder must stay frozen, just like
    # the visual feature extractor, per the assignment specification.
    for param in model.parameters():
        param.requires_grad = False

    inputs = tokenizer(
        [instruction],
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        # Use the text model directly to obtain the pooled representation.
        text_outputs = model.text_model(**inputs)
        text_features = text_outputs.pooler_output
        # Project to the shared vision-language space.
        text_features = model.text_projection(text_features)
        # L2-normalise to match CLIP's typical usage.
        text_features = F.normalize(text_features, dim=-1)

    embedding = text_features[0].cpu()  # shape: (embed_dim,)

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(embedding, cache_path)
    print(f"Saved language embedding to {cache_path}")
    return embedding


def get_language_embedding_dim(cache_path: str = _LANG_EMBED_CACHE_PATH) -> int:
    """Return the dimensionality of the cached language embedding."""
    emb = extract_language_embedding(cache_path=cache_path)
    return int(emb.shape[0])
