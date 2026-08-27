"""A from-scratch replication of SeedLM (Shafipour et al., ICLR 2025).

No official implementation was released with the paper; everything here is
written from the text, Table 1, Table 6 and Algorithms 1-3.

    >>> import numpy as np
    >>> from seedlm import compress, decompress, SEEDLM_4BIT
    >>> W = np.random.default_rng(0).normal(0, 0.02, (64, 64)).astype(np.float32)
    >>> ct = compress(W, SEEDLM_4BIT)
    >>> ct.bits_per_element()
    4.0
    >>> What = decompress(ct)
"""

from .lfsr import (
    LFSR_TAPS,
    build_U_all,
    build_V_cached,
    build_V_direct,
    full_cycle,
    is_maximal_length,
    lfsr_sequence,
    lfsr_state_for_offset,
    normalize_states,
)
from .quant import (
    QuantSpec,
    dequantize,
    quantize,
    quantize_dequantize,
    shared_exponent,
)
from .pack import pack, packed_nbytes, unpack
from .compress import (
    SEEDLM_3BIT,
    SEEDLM_4BIT,
    Codebook,
    CompressedTensor,
    SeedLMConfig,
    compress,
    decompress,
    relative_error,
)

__version__ = "0.1.0"

__all__ = [
    "LFSR_TAPS",
    "lfsr_sequence",
    "full_cycle",
    "is_maximal_length",
    "normalize_states",
    "build_V_direct",
    "build_V_cached",
    "build_U_all",
    "lfsr_state_for_offset",
    "QuantSpec",
    "shared_exponent",
    "quantize",
    "dequantize",
    "quantize_dequantize",
    "pack",
    "unpack",
    "packed_nbytes",
    "SeedLMConfig",
    "SEEDLM_3BIT",
    "SEEDLM_4BIT",
    "Codebook",
    "CompressedTensor",
    "compress",
    "decompress",
    "relative_error",
]
