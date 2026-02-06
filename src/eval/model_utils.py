import socket

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM

from src.settings import settings


def get_free_port():
    """Get a free TCP port number

    Returns:
        int: A free port number
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _load_tokenizer(path_or_id: str):
    """Load a tokenizer and set left padding

    Args:
        path_or_id (str): Model path or ID

    Returns:
        AutoTokenizer: Configured tokenizer
    """
    tok = AutoTokenizer.from_pretrained(
        path_or_id, trust_remote_code=True, token=settings.hf_token
    )
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return tok


def load_model(model_path: str, dtype=torch.bfloat16):
    """Load a model and tokenizer from local or Hub

    Args:
        model_path (str): Model path or Hub ID
        dtype: Model dtype

    Returns:
        tuple: (model, tokenizer)
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        token=settings.hf_token,
    )
    tok = _load_tokenizer(model_path)

    return model, tok


def load_vllm_model(model_path: str):
    """Load a model and tokenizer from local or Hub

    Args:
        model_path (str): Model path or Hub ID

    Returns:
        tuple: (llm, tokenizer)
    """
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0

    # Configure vLLM based on available hardware
    llm_kwargs = {
        "model": model_path,
        # vLLMでは非推奨のtorch_dtypeではなく、dtypeを使用
        "dtype": torch.bfloat16,
        "enable_prefix_caching": True,
        "max_num_seqs": 32,
        "max_model_len": 8192,
        "hf_token": settings.hf_token,
    }

    if cuda_available and device_count > 0:
        # GPU (CUDA) 構成を適用
        llm_kwargs.update(
            {
                "tensor_parallel_size": device_count,
                "gpu_memory_utilization": 0.9,
            }
        )
    else:
        # CPU専用構成を適用
        llm_kwargs.update(
            {
                "device": "cpu",
                "tensor_parallel_size": 1,  # CPUでは通常1または0（vLLMのCPUサポートに依存）
            }
        )

    llm = LLM(**llm_kwargs)

    tok = llm.get_tokenizer()
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return llm, tok
