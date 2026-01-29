from typing import Tuple

import torch

from api.ezaudio import EzAudio
from torch import Tensor

CKPT_PATH = "/data/jq_data/EzAudio-model/ckpts/s3/ezaudio_s3_xl.pt"
VAE_PATH = "/data/jq_data/EzAudio-model/ckpts/vae/1m.pt"


def load_model(
    model_file_path: str = "", model_hub: str = ""
) -> torch.nn.Module:
    """
    Returns a torch.nn.Module that produces embeddings for audio.

    Args:
        model_file_path: Ignored.
        model_hub: Which wav2vec2 model to load from hugging face.
    Returns:
        Model
    """
    model = EzAudio(model_name='s3_xl', device="cuda", ckpt_path=CKPT_PATH, vae_path=VAE_PATH)

    # sample rate and embedding sizes are required model attributes for the HEAR API
    setattr(model, "sample_rate", 24000)
    setattr(model, "embedding_size", 128)
    setattr(model, "scene_embedding_size", model.embedding_size)
    setattr(model, "timestamp_embedding_size", model.embedding_size)

    return model


def get_timestamp_embeddings(
    audio: Tensor,
    model: torch.nn.Module,
) -> Tuple[Tensor, Tensor]:
    """
    This function returns embeddings at regular intervals centered at timestamps. Both
    the embeddings and corresponding timestamps (in milliseconds) are returned.

    Args:
        audio: n_sounds x n_samples of mono audio in the range [-1, 1].
        model: Loaded model.

    Returns:
        - Tensor: embeddings, A float32 Tensor with shape (n_sounds, n_timestamps,
            model.timestamp_embedding_size).
        - Tensor: timestamps, Centered timestamps in milliseconds corresponding
            to each embedding in the output. Shape: (n_sounds, n_timestamps).
    """

    # Assert audio is of correct shape
    if audio.ndim != 2:
        raise ValueError(
            "audio input tensor must be 2D with shape (n_sounds, num_samples)"
        )

    # Make sure the correct model type was passed in

    # Send the model to the same device that the audio tensor is on.
    # model = model.to(audio.device)

    # Put the model into eval mode, and not computing gradients while in inference.
    # Iterate over all batches and accumulate the embeddings for each frame.
    if audio.ndim == 2:
        audio = audio.unsqueeze(1)  # mono to stereo
    embeddings = model.vae_encode(audio)

    total_frames = embeddings.shape[1]
    original_length = audio.shape[2] / model.sample_rate * 1000  # in milliseconds

    timestamps = torch.linspace(0, original_length, steps=total_frames + 1).unsqueeze(0)
    # get mid of each frame
    timestamps = (timestamps[:, :-1] + timestamps[:, 1:]) / 2
    
    assert timestamps.shape[1] == embeddings.shape[1]

    return embeddings, timestamps

# TODO: There must be a better way to do scene embeddings,
# e.g. just truncating / padding the audio to 2 seconds
# and concatenating a subset of the embeddings.
def get_scene_embeddings(
    audio: Tensor,
    model: torch.nn.Module,
) -> Tensor:
    """
    This function returns a single embedding for each audio clip. In this baseline
    implementation we simply summarize the temporal embeddings from
    get_timestamp_embeddings() using torch.mean().

    Args:
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in
            a batch will be padded/trimmed to the same length.
        model: Loaded model.

    Returns:
        - embeddings, A float32 Tensor with shape
            (n_sounds, model.scene_embedding_size).
    """
    embeddings, _ = get_timestamp_embeddings(audio, model)
    embeddings = torch.mean(embeddings, dim=1)
    return embeddings