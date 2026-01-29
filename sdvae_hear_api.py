import argparse
import json
import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
from tqdm import tqdm

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict

def set_seed(seed=42):
    """设置所有随机种子确保可重复性"""
    import random
    
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def set_h20():
    print("\033[1;92m❌ Using H20, disabling TF32\033[0m")
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

class AudioAutoencoderInference:
    """音频自编码器推理类"""
    
    def __init__(
        self,
        model_config_path: str,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_half: bool = False,
    ):
        """
        初始化推理器
        
        Args:
            model_config_path: 模型配置文件路径 (JSON)
            checkpoint_path: 模型权重文件路径
            device: 运行设备 ("cuda" 或 "cpu")
            use_half: 是否使用半精度 (FP16)
        """
        self.device = device
        self.use_half = use_half
        
        # 加载配置
        print(f"📄 加载模型配置: {model_config_path}")
        with open(model_config_path) as f:
            self.model_config = json.load(f)
        
        self.sample_rate = self.model_config.get("sample_rate", 48000)
        self.model_type = self.model_config.get("model_type", "autoencoder")
        
        # 创建模型
        print(f"🏗️  创建模型: {self.model_type}")
        self.model = create_model_from_config(self.model_config)
        
        # 加载权重
        print(f"📦 加载权重: {checkpoint_path}")
        state_dict = load_ckpt_state_dict(checkpoint_path)
        
        # 处理可能的包装层 (如 Lightning 的 state_dict)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        # 移除可能的前缀 (如 "model." 或 "autoencoder.")
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            for prefix in ["model.", "autoencoder.", "pretransform."]:
                if k.startswith(prefix):
                    new_key = k[len(prefix):]
                    break
            new_state_dict[new_key] = v
        
        self.model.load_state_dict(new_state_dict, strict=False)
        
        # 移到设备并设置为评估模式
        self.model = self.model.to(device)
        if use_half:
            self.model = self.model.half()
        self.model.eval()
        
        # 检查是否支持文本条件
        self.is_text_conditioned = hasattr(self.model, 'encode_text')
        
        print(f"✅ 模型加载完成!")
        print(f"   设备: {device}")
        print(f"   采样率: {self.sample_rate} Hz")
        print(f"   文本条件: {'支持' if self.is_text_conditioned else '不支持'}")
        print(f"   精度: {'FP16' if use_half else 'FP32'}")
        
        self.audio_channels = self.model_config.get("audio_channels", 2)

    def load_audio(self, audio_path: str) -> torch.Tensor:
        """
        加载音频文件并预处理
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            audio: [1, channels, samples] 的张量
        """
        audio, sr = torchaudio.load(audio_path)
        
        # 重采样到目标采样率
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # 确保是双声道
        audio_channels = self.audio_channels
        if audio.shape[0] < audio_channels:
            # 单声道转立体声
            audio = audio.repeat(audio_channels, 1)
        elif audio.shape[0] > audio_channels:
            # 多声道转目标声道数
            audio = audio[:audio_channels]
        
        # 添加batch维度: [channels, samples] -> [1, channels, samples]
        audio = audio.unsqueeze(0)
        
        return audio
    
    def save_audio(self, audio: torch.Tensor, output_path: str):
        """
        保存音频到文件
        
        Args:
            audio: [1, channels, samples] 的张量
            output_path: 输出文件路径
        """
        # 移除batch维度: [1, channels, samples] -> [channels, samples]
        audio = audio.squeeze(0).cpu()
        
        # 裁剪到 [-1, 1] 范围
        audio = torch.clamp(audio, -1.0, 1.0)
        
        # 保存
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torchaudio.save(output_path, audio, self.sample_rate)
    
    @torch.no_grad()
    def encode(
        self, 
        audio: Union[str, torch.Tensor],
        return_info: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        编码音频到潜在空间
        
        Args:
            audio: 音频路径或张量 [1, channels, samples]
            return_info: 是否返回额外信息
            
        Returns:
            latents: 潜在表示 [1, latent_dim, latent_len]
            info: (可选) 编码信息字典
        """
        # 加载音频
        if isinstance(audio, str):
            audio = self.load_audio(audio)
        
        audio = audio.to(self.device)
        if self.use_half:
            audio = audio.half()
        
        # 编码
        result = self.model.encode(audio, return_info=return_info)
        
        if return_info:
            latents, info = result
            return latents, info
        else:
            return result
    
    @torch.no_grad()
    def decode(
        self, 
        latents: torch.Tensor,
        caption: Optional[str] = None,
    ) -> torch.Tensor:
        """
        从潜在表示解码音频
        
        Args:
            latents: 潜在表示 [1, latent_dim, latent_len]
            caption: (可选) 文本描述，用于条件解码
            
        Returns:
            audio: 重建的音频 [1, channels, samples]
        """
        latents = latents.to(self.device)
        
        # 如果是文本条件模型且提供了caption
        if self.is_text_conditioned and caption is not None:
            # 编码文本
            text_embeds, attention_mask = self.model.encode_text([caption])
            
            # 条件解码
            audio = self.model.decode(
                latents, 
                text_embeds=text_embeds,
                text_attention_mask=attention_mask
            )
        else:
            # 无条件解码
            audio = self.model.decode(latents)
        
        return audio
    
    @torch.no_grad()
    def reconstruct(
        self, 
        audio: Union[str, torch.Tensor],
        caption: Optional[str] = None,
        return_latents: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        重建音频 (编码 + 解码)
        
        Args:
            audio: 音频路径或张量
            caption: (可选) 文本描述
            return_latents: 是否同时返回潜在表示
            
        Returns:
            reconstructed: 重建的音频
            latents: (可选) 潜在表示
        """
        # 编码
        latents = self.encode(audio)
        
        # 解码
        reconstructed = self.decode(latents, caption=caption)
        
        if return_latents:
            return reconstructed, latents
        else:
            return reconstructed
    
    def inference_single(
        self,
        input_path: str,
        output_path: str,
        caption: Optional[str] = None,
        idx: int = None,
    ):
        """
        单个文件推理
        
        Args:
            input_path: 输入音频路径
            output_path: 输出音频路径
            caption: (可选) 文本描述
        """
           
        # 加载音频
        audio = self.load_audio(input_path)
          
        # 重建
        reconstructed, latents = self.reconstruct(audio, caption=caption, return_latents=True)
        
        
        # 保存
        self.save_audio(reconstructed, output_path)
        if idx is not None and idx<5:
            print(f"   索引: {idx}")
            print(f"🎵 处理: {input_path}")
            print(f"   原始音频: {audio.shape}")
            print(f"   潜在表示: {latents.shape}")
            print(f"   重建音频: {reconstructed.shape}")
            print(f"💾 保存到: {output_path}")
    
    def inference_batch(
        self,
        input_dir: str,
        output_dir: str,
        caption: Optional[str] = None,
        extensions: tuple = ('.wav', '.mp3', '.flac', '.ogg'),
    ):
        """
        批量推理
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            caption: (可选) 统一的文本描述
            extensions: 支持的音频格式
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 收集所有音频文件
        audio_files = []
        for ext in extensions:
            audio_files.extend(input_dir.glob(f"*{ext}"))
        
        print(f"📁 找到 {len(audio_files)} 个音频文件")
        
        # 批量处理
        for idx, audio_file in enumerate(tqdm(audio_files, desc="推理进度")):
            output_file = output_dir / f"{audio_file.stem}.wav"

            self.inference_single(str(audio_file), str(output_file), caption=caption, idx=idx)
            


from diffusers.models.autoencoders.autoencoder_oobleck import AutoencoderOobleck
import torch
from torch import nn
from typing import List
import argparse
import os
import torchaudio
from pathlib import Path
import copy
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Optional, Union, List
from math import pi
import yaml
from safetensors.torch import load_file



class VAEInference(AudioAutoencoderInference):

    def __init__(
        self,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_half=False,
        sample_rate=44100,
    ):

        self.model = AutoencoderOobleck.from_pretrained(
            "/data/jq_data/stable-audio-open-1.0", subfolder="vae"
        )

        self.model.to(device)
        self.device = device
        self.use_half = use_half
        self.sample_rate = sample_rate
        self.audio_channels = 2


    @torch.no_grad()
    def encode(
        self, 
        audio: Union[str, torch.Tensor],
        return_info: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        Encode audio to latent space
        
        Args:
            audio: Audio path or tensor [1, channels, samples]
            return_info: Whether to return extra info
            
        Returns:
            latents: Latent representation [1, latent_dim, latent_len]
            info: (optional) Encoding info dict
        """
        assert return_info == False, "return_info is not supported"
        # 加载音频
        if isinstance(audio, str):
            audio = self.load_audio(audio)
        
        audio = audio.to(self.device)
        if self.use_half:
            audio = audio.half()
        
        # 编码
        result = self.model.encode(audio)
        result = result.latent_dist.sample()

        return result

    @torch.no_grad()
    def decode(
        self, 
        latents: torch.Tensor,
        caption: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Decode audio from latent representation
        
        Args:
            latents: Latent representation [1, latent_dim, latent_len]
            caption: (optional) Text description for conditional decoding
            
        Returns:
            audio: Reconstructed audio [1, channels, samples]
        """
        latents = latents.to(self.device)
        
      
        audio = self.model.decode(latents).sample.cpu()[0]
        
        return audio

        

def main():
    parser = argparse.ArgumentParser(description="Audio autoencoder inference")
    
    # Model parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda/cpu)")
    parser.add_argument("--half", action="store_true",
                        help="Use half precision (FP16)")
    
    # Input/Output
    parser.add_argument("--input", type=str,
                        help="Input audio file path (single file mode)")
    parser.add_argument("--output", type=str,
                        help="Output audio file path (single file mode)")
    parser.add_argument("--input_dir", type=str, default="./input",
                        help="Input directory path (batch mode)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory path (batch mode)")
    
    # Condition parameters
    parser.add_argument("--caption", type=str, default=None,
                        help="Text description (for conditional generation)")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)

    # 创建推理器
    inferencer = VAEInference(
        device=args.device,
        use_half=args.half,
    )

    encode_output = inferencer.encode(torch.randn(1, 2, 20*44100), return_info=False)
    encode_output = encode_output.transpose(-1,-2)
    print("Encode output shape:", encode_output.shape)
    
    # # 执行推理
    # if args.input and args.output:
    #     # 单文件模式
    #     inferencer.inference_single(
    #         input_path=args.input,
    #         output_path=args.output,
    #         caption=args.caption,
    #     )
    # elif args.input_dir and args.output_dir:
    #     # 批量模式
    #     inferencer.inference_batch(
    #         input_dir=args.input_dir,
    #         output_dir=args.output_dir,
    #         caption=args.caption,
    #     )
    # else:
    #     print("Error: Please specify --input/--output or --input_dir/--output_dir")
    #     parser.print_help()




from typing import Tuple

import torch

from torch import Tensor

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
    # 设置随机种子
    set_seed(42)

    # 创建推理器
    model = VAEInference(
        device="cuda",
    )

    # sample rate and embedding sizes are required model attributes for the HEAR API
    setattr(model, "sample_rate", 44100)
    setattr(model, "embedding_size", 64)
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
    
    # convert audio to [n_sounds, channels, n_samples] where channels=2
    if audio.ndim == 2:
        audio = audio.unsqueeze(1).repeat(1, 2, 1)  # mono to stereo

    embeddings = model.encode(audio, return_info=False)
    embeddings = embeddings.transpose(-1,-2)

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