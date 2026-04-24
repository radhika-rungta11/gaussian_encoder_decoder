import numpy as np

def quantize_scalar(data, num_bits=8):
    """
    Min-max quantize a scalar array to an unsigned integer dtype.
    Returns the quantized array and a float32 [min, max] range.
    """
    data = np.asarray(data, dtype=np.float32)
    if data.size == 0:
        return np.zeros_like(data, dtype=np.uint8), np.array([0.0, 0.0], dtype=np.float32)

    min_val = float(np.min(data))
    max_val = float(np.max(data))
    max_quantized_val = (1 << num_bits) - 1

    if max_val - min_val < 1e-12:
        quantized = np.zeros_like(data, dtype=np.uint16 if num_bits > 8 else np.uint8)
    else:
        normalized = (data - min_val) / (max_val - min_val)
        quantized = np.rint(normalized * max_quantized_val)
        dtype = np.uint16 if num_bits > 8 else np.uint8
        quantized = np.clip(quantized, 0, max_quantized_val).astype(dtype)

    return quantized, np.array([min_val, max_val], dtype=np.float32)


def quantize_vector(data, num_bits=8):
    """
    Min-max quantize a vector array independently per channel.
    Returns:
      - quantized array
      - min values, shape (D,)
      - max values, shape (D,)
    """
    data = np.asarray(data, dtype=np.float32)
    if data.size == 0:
        quantized = np.zeros_like(data, dtype=np.uint8)
        empty = np.zeros((data.shape[-1],), dtype=np.float32) if data.ndim > 1 else np.zeros((1,), dtype=np.float32)
        return quantized, empty, empty

    if data.ndim == 1:
        data = data[:, None]

    min_vals = np.min(data, axis=0).astype(np.float32)
    max_vals = np.max(data, axis=0).astype(np.float32)
    span = np.maximum(max_vals - min_vals, 1e-12)
    max_quantized_val = (1 << num_bits) - 1

    normalized = (data - min_vals[None, :]) / span[None, :]
    quantized = np.rint(normalized * max_quantized_val)
    dtype = np.uint16 if num_bits > 8 else np.uint8
    quantized = np.clip(quantized, 0, max_quantized_val).astype(dtype)
    return quantized, min_vals, max_vals


def inverse_quantize_scalar(quantized_data, min_val, max_val, num_bits=8):
    """
    Inverse min-max quantization for scalar properties (e.g. opacity)
    quantized_data: array of integers (e.g. uint8)
    min_val: float
    max_val: float
    num_bits: number of bits used for quantization (typically 8 for uint8)
    """
    quantized_data = np.asarray(quantized_data, dtype=np.float32)
    max_quantized_val = (1 << num_bits) - 1
    
    normalized = quantized_data / max_quantized_val
    return min_val + normalized * (max_val - min_val)

def inverse_quantize_vector(quantized_data, min_vals, max_vals, num_bits=8):
    """
    Inverse min-max quantization for vector properties (e.g. scale, rotation)
    min_vals, max_vals: arrays matching the last dimension of quantized_data.
    """
    quantized_data = np.asarray(quantized_data, dtype=np.float32)
    squeeze_last = False
    if quantized_data.ndim == 1:
        quantized_data = quantized_data[:, None]
        squeeze_last = True

    min_vals = np.asarray(min_vals, dtype=np.float32).reshape(1, -1)
    max_vals = np.asarray(max_vals, dtype=np.float32).reshape(1, -1)
    
    max_quantized_val = (1 << num_bits) - 1
    normalized = quantized_data / max_quantized_val
    decoded = min_vals + normalized * (max_vals - min_vals)
    if squeeze_last:
        return decoded[:, 0]
    return decoded

def decode_rvq(indices, codebook):
    """
    Residual Vector Quantization (RVQ) decoding stub.
    indices: array of shape (N, num_stages) giving the codebook index at each stage.
    codebook: array of shape (num_stages, codebook_size, feature_dim)
    
    TODO: The exact shapes of the codebook and indices depend on the provided NPZ.
    """
    indices = np.asarray(indices, dtype=np.int32)
    # Sum the vectors from each stage's codebook
    N, num_stages = indices.shape
    
    if codebook.ndim == 2:
        # Simple VQ (single stage)
        return codebook[indices]
    
    output = np.zeros((N, codebook.shape[-1]), dtype=np.float32)
    for stage in range(num_stages):
        output += codebook[stage, indices[:, stage]]
    return output
