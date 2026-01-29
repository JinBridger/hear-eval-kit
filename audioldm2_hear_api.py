#!/usr/bin/python3
import os
import torch
import logging
import torchaudio
import soundfile as sf
import argparse
from tqdm import tqdm
import json
from torch import Tensor
from typing import Tuple

from audioldm2 import build_model
from audioldm2.utilities.audio import TacotronSTFT
from audioldm2.utilities.audio.tools import wav_to_fbank, _pad_spec, get_mel_from_wav
from audioldm2.utils import default_audioldm_config

os.environ["TOKENIZERS_PARALLELISM"] = "true"
MODEL_NAME = "audioldm2-full"


def seed_everything(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(42)

# Get STFT config
config = default_audioldm_config(MODEL_NAME)
fn_STFT = TacotronSTFT(
    config["preprocessing"]["stft"]["filter_length"],
    config["preprocessing"]["stft"]["hop_length"],
    config["preprocessing"]["stft"]["win_length"],
    config["preprocessing"]["mel"]["n_mel_channels"],
    config["preprocessing"]["audio"]["sampling_rate"],
    config["preprocessing"]["mel"]["mel_fmin"],
    config["preprocessing"]["mel"]["mel_fmax"],
)
fn_STFT = fn_STFT

sample_rate = config["preprocessing"]["audio"]["sampling_rate"]
print("Sample rate:", sample_rate)

# Output path


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
    audioldm = build_model(model_name=MODEL_NAME, device="cuda")

    # Get VAE (first_stage_model) and vocoder
    model = audioldm.first_stage_model
    model.eval()

    # sample rate and embedding sizes are required model attributes for the HEAR API
    setattr(model, "sample_rate", sample_rate)
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
    model.eval()
    with torch.no_grad():
        normalize=False
        duration=audio.shape[1] / model.sample_rate
        embeddings = None
        # for each audio in the batch
        for i in range(audio.shape[0]):
            waveform = audio[i, ...]
            waveform = waveform.cpu().numpy()
            waveform = torch.FloatTensor(waveform)

            fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)

            fbank = torch.FloatTensor(fbank.T)
            log_magnitudes_stft = torch.FloatTensor(log_magnitudes_stft.T)

            fbank, log_magnitudes_stft = _pad_spec(fbank, int(duration * 102.4)), _pad_spec(
                log_magnitudes_stft, int(duration * 102.4)
            )

            mel = fbank.unsqueeze(0).unsqueeze(0).to("cuda")  # [1, 1, T, n_mels]
            
            # Encode
            posterior = model.encode(mel)
            embed = posterior.mean.transpose(1, 2)
            embedding = embed.reshape(embed.size(0), embed.size(1), -1)
            if embeddings is None:
                embeddings = embedding
            else:
                embeddings = torch.cat((embeddings, embedding), dim=0)


    total_frames = embeddings.shape[1]
    original_length = audio.shape[1] / model.sample_rate * 1000  # in milliseconds

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