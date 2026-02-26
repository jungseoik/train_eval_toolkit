"""
Training import smoke tests.

목적: training에 필요한 패키지 import가 모두 정상인지 확인.
실행: PYTHONPATH="$(pwd)" pytest tests/test_imports.py -v
"""

import pytest
from tests.utils.import_helpers import skip_if_missing


# ---------------------------------------------------------------------------
# 1. PyTorch
# ---------------------------------------------------------------------------

def test_torch():
    import torch
    assert torch.__version__.startswith("2.1"), f"Expected 2.1.x, got {torch.__version__}"


def test_torchvision():
    import torchvision
    assert torchvision.__version__.startswith("0.16"), f"Expected 0.16.x, got {torchvision.__version__}"


def test_torch_distributed():
    import torch.distributed


def test_torchvision_transforms():
    from torchvision.transforms.functional import InterpolationMode


# ---------------------------------------------------------------------------
# 2. Transformers 스택
# ---------------------------------------------------------------------------

def test_transformers():
    import transformers
    assert transformers.__version__.startswith("4.37"), f"Expected 4.37.x, got {transformers.__version__}"


def test_transformers_trainer():
    from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed


def test_transformers_auto():
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def test_peft():
    from peft import LoraConfig, get_peft_model


def test_tokenizers():
    from tokenizers import Tokenizer, decoders, normalizers, processors
    from tokenizers.models import BPE


def test_sentencepiece():
    import sentencepiece as spm


def test_accelerate():
    import accelerate
    assert accelerate.__version__.startswith("0.28"), f"Expected 0.28.x, got {accelerate.__version__}"


def test_datasets():
    import datasets


# ---------------------------------------------------------------------------
# 3. Distributed Training
# ---------------------------------------------------------------------------

def test_deepspeed():
    import deepspeed
    assert deepspeed.__version__.startswith("0.13"), f"Expected 0.13.x, got {deepspeed.__version__}"


# ---------------------------------------------------------------------------
# 4. GPU-dependent (skip if not installed)
# ---------------------------------------------------------------------------

@skip_if_missing("flash_attn")
def test_flash_attn():
    from flash_attn import __version__ as flash_attn_version
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func


@skip_if_missing("bitsandbytes")
def test_bitsandbytes():
    import bitsandbytes


# ---------------------------------------------------------------------------
# 5. Vision
# ---------------------------------------------------------------------------

def test_timm():
    import timm
    from timm.models.layers import DropPath


def test_einops():
    from einops import rearrange


def test_einops_exts():
    import einops_exts


def test_PIL():
    from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError


def test_opencv():
    import cv2


def test_imageio():
    import imageio


def test_decord():
    from decord import VideoReader


# ---------------------------------------------------------------------------
# 6. Data / Utilities
# ---------------------------------------------------------------------------

def test_numpy():
    import numpy as np
    assert np.__version__.startswith("1.26"), f"Expected 1.26.x, got {np.__version__}"


def test_scipy():
    import scipy


def test_sklearn():
    import sklearn


def test_orjson():
    import orjson


def test_yaml():
    import yaml


def test_yacs():
    from yacs.config import CfgNode


def test_tqdm():
    import tqdm


def test_shortuuid():
    import shortuuid


def test_tensorboardX():
    import tensorboardX


def test_termcolor():
    import termcolor


def test_pycocoevalcap():
    import pycocoevalcap


# ---------------------------------------------------------------------------
# 7. 내부 모듈 (src.training.internvl)
# ---------------------------------------------------------------------------

def test_src_internvl_model():
    try:
        from src.training.internvl.model.internvl_chat import (
            InternVisionConfig,
            InternVisionModel,
            InternVLChatConfig,
            InternVLChatModel,
        )
    except ImportError as e:
        pytest.skip(f"PYTHONPATH 미설정 또는 flash_attn 미설치: {e}")


def test_src_internvl_dist_utils():
    try:
        from src.training.internvl.dist_utils import init_dist
    except ImportError as e:
        pytest.skip(str(e))


def test_src_internvl_conversation():
    try:
        from src.training.internvl.conversation import get_conv_template
    except ImportError as e:
        pytest.skip(str(e))


def test_src_internvl_patch():
    try:
        from src.training.internvl.patch import (
            concat_pad_data_collator,
            replace_llama_attention_class,
            replace_train_dataloader,
        )
    except ImportError as e:
        pytest.skip(f"flash_attn 미설치로 skip: {e}")


def test_src_internvl_train_constants():
    try:
        from src.training.internvl.train.constants import (
            IMG_CONTEXT_TOKEN,
            IMG_END_TOKEN,
            IMG_START_TOKEN,
        )
    except ImportError as e:
        pytest.skip(str(e))
