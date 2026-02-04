import os
import re
import socket
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import setup_credentials

# Set up credentials and environment
config = setup_credentials()


def get_free_port():
    """未使用のTCPポート番号を取得する

    Returns:
        int: 利用可能なポート番号
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ---------- 工具 ----------
_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")


def _pick_latest_checkpoint(model_path: str) -> str:
    """チェックポイント群から最新のものを選択

    Args:
        model_path (str): モデルディレクトリのパス

    Returns:
        str: 最新チェックポイントのパス（なければ入力を返す）
    """
    ckpts = [
        (int(m.group(1)), p)
        for p in Path(model_path).iterdir()
        if (m := _CHECKPOINT_RE.fullmatch(p.name)) and p.is_dir()
    ]
    return str(max(ckpts, key=lambda x: x[0])[1]) if ckpts else model_path


def _is_lora(path: str) -> bool:
    """指定パスがLoRAアダプタを含むか判定

    Args:
        path (str): 対象パス

    Returns:
        bool: LoRAアダプタ構成を含む場合True
    """
    return Path(path, "adapter_config.json").exists()


def _load_and_merge_lora(lora_path: str, dtype, device_map):
    """LoRAアダプタをベースモデルへマージして読み込む

    Args:
        lora_path (str): LoRAディレクトリ
        dtype: モデルのdtype
        device_map: デバイス配置

    Returns:
        AutoModelForCausalLM: マージ済みモデル
    """
    cfg = PeftConfig.from_pretrained(lora_path, token=config.hf_token)

    # Gemma 2モデルかどうかをチェック
    is_gemma2 = "gemma-2" in cfg.base_model_name_or_path.lower()

    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": True,
        "token": config.hf_token,
    }
    # Gemma 2の場合はFlash Attentionを無効化
    if is_gemma2:
        model_kwargs["attn_implementation"] = "eager"
        print("Gemma 2モデルが検出されました。Flash Attentionを無効化します。")

    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name_or_path, **model_kwargs
    )
    return PeftModel.from_pretrained(
        base, lora_path, token=config.hf_token
    ).merge_and_unload()


def _load_tokenizer(path_or_id: str):
    """トークナイザーを読み込み、左パディングに設定

    Args:
        path_or_id (str): モデルパスまたはID

    Returns:
        AutoTokenizer: 設定済みトークナイザー
    """
    tok = AutoTokenizer.from_pretrained(
        path_or_id, trust_remote_code=True, token=config.hf_token
    )
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return tok


def load_model(model_path: str, dtype=torch.bfloat16):
    """ローカルまたはHubからモデルとトークナイザーを読み込む

    LoRA構成があればマージして返す。

    Args:
        model_path (str): モデルのローカルパスまたはHub ID
        dtype: モデルのdtype

    Returns:
        tuple: (model, tokenizer)
    """
    # Gemma 2モデルかどうかをチェック
    is_gemma2 = "gemma-2" in model_path.lower()

    if not os.path.exists(model_path):  # ---- Hub ----
        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto",
            "trust_remote_code": True,
            "token": config.hf_token,
        }
        # Gemma 2の場合はFlash Attentionを無効化
        if is_gemma2:
            model_kwargs["attn_implementation"] = "eager"
            print("Gemma 2モデルが検出されました。Flash Attentionを無効化します。")
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        tok = _load_tokenizer(model_path)
        return model, tok

    resolved = _pick_latest_checkpoint(model_path)
    print(f"loading {resolved}")
    if _is_lora(resolved):
        model = _load_and_merge_lora(resolved, dtype, "auto")
        tok = _load_tokenizer(model.config._name_or_path)
    else:
        # ローカルパスもGemma 2かチェック
        if not is_gemma2:
            is_gemma2 = "gemma-2" in resolved.lower()

        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto",
            "trust_remote_code": True,
            "token": config.hf_token,
        }
        # Gemma 2の場合はFlash Attentionを無効化
        if is_gemma2:
            model_kwargs["attn_implementation"] = "eager"
            print("Gemma 2モデルが検出されました。Flash Attentionを無効化します。")
        model = AutoModelForCausalLM.from_pretrained(resolved, **model_kwargs)
        tok = _load_tokenizer(resolved)
    return model, tok


def load_vllm_model(model_path: str):
    """vLLMバックエンドでモデルを読み込み、必要に応じてLoRAを有効化

    Args:
        model_path (str): モデルのローカルパスまたはHub ID

    Returns:
        tuple: (llm, tokenizer, lora_path or None)
    """
    from vllm import LLM

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0

    # Gemma 2モデルかどうかをチェック
    is_gemma2 = "gemma-2" in model_path.lower()

    if not os.path.exists(model_path):  # ---- Hub ----
        # Configure vLLM based on available hardware
        llm_kwargs = {
            "model": model_path,
            # vLLMでは非推奨のtorch_dtypeではなく、dtypeを使用
            "dtype": torch.bfloat16,
            "enable_prefix_caching": True,
            "max_num_seqs": 32,
            "max_model_len": 8192,
            "hf_token": config.hf_token,
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

        if is_gemma2:
            llm_kwargs["enforce_eager"] = True
            print(
                "Gemma 2モデルが検出されました: Flash Attention (FA) のRuntime Error回避のため、Eager Modeを強制 (FAを無効化) します。"
            )

        llm = LLM(**llm_kwargs)

        tok = llm.get_tokenizer()
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "left"
        return llm, tok, None

    resolved = _pick_latest_checkpoint(model_path)
    print(f"loading {resolved}")
    is_lora = _is_lora(resolved)

    base_path = (
        PeftConfig.from_pretrained(resolved).base_model_name_or_path
        if is_lora
        else resolved
    )

    # ベースパスもGemma 2かチェック
    if not is_gemma2:
        is_gemma2 = "gemma-2" in base_path.lower()

    llm_kwargs = {
        "model": base_path,
        # vLLMのベストプラクティスに従い、dtypeを指定 (bfloat16はGPUに適していることが多い)
        "dtype": torch.bfloat16,
        "enable_prefix_caching": True,
        "enable_lora": True,
        "max_num_seqs": 32,
        "max_model_len": 20000,
        "max_lora_rank": 128,
        "hf_token": config.hf_token,
    }

    # --- ハードウェアに基づく設定の適用 ---
    if cuda_available and device_count > 0:
        # GPU (CUDA) 構成: Tensor Parallelism とメモリ使用率を設定
        llm_kwargs.update(
            {
                "tensor_parallel_size": device_count,
                "gpu_memory_utilization": 0.9,
            }
        )
    else:
        # CPU専用構成: デバイスとTensor Parallelismを設定
        llm_kwargs.update(
            {
                "device": "cpu",
                # CPUでは通常、並列処理は使わないか、コア数に応じて設定（vLLMのCPUサポートに依存）
                "tensor_parallel_size": 1,
            }
        )

    # --- モデル固有の互換性修正 (Gemma 2) の適用 ---
    # Flash Attention (FA) の "tanh softcapping" によるRuntime Error回避のため、Eager Modeを強制
    if is_gemma2:
        llm_kwargs["enforce_eager"] = True
        print(
            "✅ Gemma 2モデルが検出されました: FAのRuntime Error回避のため、Eager Modeを強制 (FAを無効化) します。"
        )

    # LLMの初期化
    llm = LLM(**llm_kwargs)

    if is_lora:
        lora_path = resolved
    else:
        lora_path = None

    tok = llm.get_tokenizer()
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return llm, tok, lora_path
