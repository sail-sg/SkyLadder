"""Full definition of a GPT NeoX Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""
import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self
from flash_attn import flash_attn_func
from lit_gpt.config import Config
from xformers.ops import SwiGLU
from .fused_rotary_embedding import apply_rotary_emb_func
RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[KVCache] = []

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        # GPT-NeoX  https://arxiv.org/pdf/2204.06745.pdf
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            # RWKV: set it to 1e-4
            # torch.nn.init.uniform_(module.weight,  -1e-4, 1e-4)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # GPT-NeoX       
        for name, p in module.named_parameters():
            if (name == "proj.weight" and isinstance(module, LLaMAMLP)) or (name == "w3.weight" and isinstance(module, SwiGLU) or (name=="proj.weight" and isinstance(module, CausalSelfAttention))):  #if use xformer swiglu, fc2 layer will be renamed to w3
                nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(self.config.n_embd)  /  n_layer)
        

    def reset_cache(self) -> None:
        self.kv_caches.clear()
        if self.mask_cache is not None and self.mask_cache.device.type == "xla":
            # https://github.com/Lightning-AI/lit-gpt/pull/83#issuecomment-1558150179
            self.rope_cache = None
            self.mask_cache = None

    def reset_rope_cache(self,new_base) -> None:
        self.config.rope_base = new_base
        print("Resetting rope cache with new base: ", new_base)
        self.rope_cache = None

    def forward(
        self, idx: torch.Tensor, fragment_lens = None, fragment_nums = None, max_seq_length: Optional[int] = None, input_pos: Optional[torch.Tensor] = None, force_use_masking=False, window_size: Optional[int] = None
    ) -> torch.Tensor:
        if force_use_masking or (self.config.intradoc_mask and self.training):
            real_fragment_lens = [torch.tensor([0], dtype=torch.int32, device=fragment_lens.device)]
            for padded_fragment_lens, cur_fragment_num in zip(fragment_lens, fragment_nums):
                real_fragment_lens.append(padded_fragment_lens[:cur_fragment_num])
            real_fragment_lens = torch.cat(real_fragment_lens)
            max_seqlen = real_fragment_lens.max().item()
            cu_seqlens = torch.cumsum(real_fragment_lens, 0, dtype=torch.int32)
            # print(f"max_seqlen shape: {max_seqlen}, cu_seqlens shape: {cu_seqlens.shape}")
        else:
            max_seqlen, cu_seqlens = None, None

        B, T = idx.size()
        use_kv_cache = input_pos is not None

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        if use_kv_cache:  # not relevant otherwise
            assert (
                max_seq_length >= T
            ), f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert block_size >= T, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        # passing `attn_mask` to SDPA downgrades it to use the inefficient implementation. since we only need the mask
        # for the kv-cache support (only during inference), we only create it in that situation
        # this will be resolved by https://github.com/pytorch/pytorch/issues/96099
        if use_kv_cache and self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)

        cos, sin = self.rope_cache
        if use_kv_cache:

            cos = cos.index_select(0, input_pos)
            sin = sin.index_select(0, input_pos)
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            cos = cos[:T]
            sin = sin[:T]
            mask = None

        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        if not use_kv_cache:
            for block in self.transformer.h:
                assert (self.training and (cu_seqlens is not None or not self.config.intradoc_mask)) or not self.training, "cu_seqlens must be provided for intradoc mask[0]"
                x, *_ = block(x, (cos, sin), max_seq_length, cuseq_lens=cu_seqlens, max_seqlen=max_seqlen, force_use_masking=force_use_masking, window_size=window_size)
        else:
            self.kv_caches = self.kv_caches or self.build_kv_caches(x, max_seq_length, cos.size(-1) * 2)
            for i, block in enumerate(self.transformer.h):
                assert (self.training and (cu_seqlens is not None or not self.config.intradoc_mask)) or not self.training, "cu_seqlens must be provided for intradoc mask[0]"
                x, self.kv_caches[i] = block(x, (cos, sin), max_seq_length, mask, input_pos, self.kv_caches[i], cuseq_lens=cu_seqlens, max_seqlen=max_seqlen, force_use_masking=force_use_masking, window_size=window_size)
        x = self.transformer.ln_f(x)

        return self.lm_head(x)  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
        print("Building rope cache, base: ", self.config.rope_base)
        if 'rerope' not in self.config.intradoc_mask:
            return build_rope_cache(
                seq_len=self.config.block_size,
                n_elem=int(self.config.rotary_percentage * self.config.head_size),
                dtype=torch.bfloat16,
                device=idx.device,
                condense_ratio=self.config.condense_ratio,
                base=self.config.rope_base,
            )
        else:
            if self.config.intradoc_mask == 'fix2rerope':
                rerope_len = 2048
            elif self.config.intradoc_mask == 'fix1rerope':
                rerope_len = 1024
            else:
                raise NotImplementedError("Only fix2rerope and fix1rerope are supported")

            print("Building rope cache with {} length".format(rerope_len), "because of intradoc_mask: ", self.config.intradoc_mask)
            cos, sin = build_rope_cache(
                seq_len=rerope_len,
                n_elem=int(self.config.rotary_percentage * self.config.head_size),
                dtype=torch.bfloat16,
                device=idx.device,
                condense_ratio=self.config.condense_ratio,
                base=self.config.rope_base,)
            print("Original cos shape: ", cos.shape, "Original sin shape: ", sin.shape)
            factor = int(self.config.block_size // rerope_len)
            print("Factor: ", factor)
            sin = sin.repeat(factor,1)
            cos = cos.repeat(factor,1)
            print("New cos shape: ", cos.shape, "New sin shape: ", sin.shape)
            return cos, sin

    def build_mask_cache(self, idx: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((self.config.block_size, self.config.block_size), device=idx.device, dtype=torch.bool)
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def build_kv_caches(self, idx: torch.Tensor, max_seq_length: int, rope_cache_length: int) -> List[KVCache]:
        B = idx.size(0)
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_query_groups

        k_cache_shape = (
            B,
            max_seq_length,
            heads,
            rope_cache_length + self.config.head_size - int(self.config.rotary_percentage * self.config.head_size),
        )
        v_cache_shape = (B, max_seq_length, heads, self.config.head_size)
        device = idx.device
        return [
            (torch.zeros(k_cache_shape, device=device), torch.zeros(v_cache_shape, device=device))
            for _ in range(self.config.n_layer)
        ]


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)
        self.config = config
    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        cuseq_lens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        force_use_masking=False,
        window_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:

        n_1 = self.norm_1(x)
        h, new_kv_cache = self.attn(n_1, rope, max_seq_length, mask, input_pos, kv_cache, cuseq_lens=cuseq_lens, max_seqlen=max_seqlen, force_use_masking=force_use_masking, window_size=window_size)
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            x = x + h + self.mlp(n_2)
        else:
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm)."
                )

            x = x + h
            x = x + self.mlp(self.norm_2(x))
        return x, new_kv_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.config = config
        self._use_pos_emb = not (config.positional_embedding == "no")
        if not self._use_pos_emb:
            print("Warning: not using positional embeddings, got config.positional_embedding == 'no'")
        self._mask_attn = config.intradoc_mask

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        cuseq_lens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        force_use_masking=False,
        window_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size) # (B, T, n_query_groups, total_qkv, hs)
        # qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)

        # repeat k and v if necessary
        # Peiyuan: we do not need to do this as flash attention 2 already support GQA
        # if self.config.n_query_groups != 1:  # doing this would require a full kv cache with MQA (inefficient!)
        #     # for MHA this is a no-op
        #     k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
        #     v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B,  T, -1, self.config.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B,  T, -1, self.config.head_size)
        v = v.reshape(B,  T, -1, self.config.head_size)

        cos, sin = rope

        # apply rope in fp32 significanly stabalize training
        # fused rope expect (batch_size, seqlen, nheads, headdim)
        if self._use_pos_emb:
            q = apply_rotary_emb_func(q, cos, sin, False, True)
            k = apply_rotary_emb_func(k, cos, sin, False, True)
        # n_elem = int(self.config.rotary_percentage * self.config.head_size)

        # q_roped = apply_rope(q[..., :n_elem], cos.repeat(1,2), sin.repeat(1,2))
        # k_roped = apply_rope(k[..., :n_elem], cos.repeat(1,2), sin.repeat(1,2))
        # print( (q_roped - q).sum())
        # q = torch.cat((q_roped, q[..., n_elem:]), dim=-1)
        # k = torch.cat((k_roped, k[..., n_elem:]), dim=-1)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k, cache_v = cache_k.to(dtype=k.dtype), cache_v.to(dtype=v.dtype)
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=1)
                cache_v = torch.roll(cache_v, -1, dims=1)

            k = cache_k.index_copy_(1, input_pos, k)
            v = cache_v.index_copy_(1, input_pos, v)
            kv_cache = k, v

        y = self.scaled_dot_product_attention(q, k, v, mask=mask, cuseq_lens=cuseq_lens, max_seqlen=max_seqlen, force_use_masking=force_use_masking, window_size=window_size)
        y = y.reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y, kv_cache

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None, cuseq_lens: Optional[torch.Tensor] = None,
            max_seqlen: Optional[int] = None, force_use_masking=False, window_size: Optional[int] = None
    ):
        scale = 1.0 / math.sqrt(self.config.head_size)

        if (
            FlashAttention2Available
            and mask is None
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            from flash_attn import flash_attn_func, flash_attn_varlen_func
            if force_use_masking or (self._mask_attn and self.training):
                assert cuseq_lens is not None, "cu_seqlens must be provided for intradoc mask"
                assert max_seqlen is not None, "max_seqlen must be provided for intradoc mask"
                #print(f"I am here using flash attention with document mask! and I am in training mode: {self.training}")
                # print(cuseq_lens.shape, max_seqlen)
                # merge the first two dimensions of q k v
                bsize, seqlen, nhead, head_dim = q.shape
                q = q.reshape(-1, q.shape[-2], q.shape[-1])
                k = k.reshape(-1, k.shape[-2], k.shape[-1])
                v = v.reshape(-1, v.shape[-2], v.shape[-1])
                #print("New shapes", q.shape, k.shape, v.shape)
                # print("cuseq_lens", cuseq_lens, cuseq_lens.shape)
                # print("Max seqlen", max_seqlen)
                # assert window_size is None, "Window size is not supported with flash attention var len"
                if window_size is not None or self.config.window_size != -1:
                    window_size = (window_size - 1, 0) if window_size is not None else (self.config.window_size - 1, 0)
                    # print("Using window size: ", window_size)
                    # print(f"I am here using flash attention varlen with window sizeni: {window_size}")
                    result =  flash_attn_varlen_func(q, k, v, cu_seqlens_q=cuseq_lens, cu_seqlens_k=cuseq_lens,
                                                  max_seqlen_q=max_seqlen,
                                                  max_seqlen_k=max_seqlen, dropout_p=0.0, softmax_scale=scale, causal=True, window_size=window_size)
                else:
                    result = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cuseq_lens, cu_seqlens_k=cuseq_lens,
                                              max_seqlen_q=max_seqlen,
                                              max_seqlen_k=max_seqlen, dropout_p=0.0, softmax_scale=scale, causal=True)
                result = result.reshape(bsize, seqlen, nhead, head_dim)
                return result
            elif self.config.window_size != -1 or window_size is not None:
                # the input window size will overwrite the config window size
                # print("Input window size: ", window_size, "Config window size: ", self.config.window_size)
                window_size = (self.config.window_size - 1, 0) if window_size is None else (window_size - 1, 0)
                # print("Using window size: ", window_size)
                # print(f"I am here using flash attention with window size: {self.config.window_size}")
                return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True, window_size=window_size)
            else:
                return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
             k = k.repeat_interleave(q.shape[1]//k.shape[1], dim=1)
             v = v.repeat_interleave(q.shape[1]//v.shape[1], dim=1)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        # self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        # self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        # self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.swiglu = SwiGLU(config.n_embd,config.intermediate_size, bias=False, _pack_weights=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_fc_1 = self.fc_1(x)
        # x_fc_2 = self.fc_2(x)
        # x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        # return self.proj(x)
        return self.swiglu(x)


def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000, condense_ratio: int = 1
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    # added by peiyuan to ensure same data type with q, k, to use fused rotary embedding
    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.type_as(x)
