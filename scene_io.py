import gzip
import json
import math
import os
from pathlib import Path

import numpy as np
from dahuffman.huffmancodec import PrefixCodec

from decoder import SceneProperties, decode_scene
from quantize import decode_rvq


FOURDGS_MAGIC = b"4DGS"
FOURDGS_HEADER_SIZE = 12


def load_scene_properties_from_npz(npz_path, decode_auxiliary_properties=False):
    if decode_auxiliary_properties:
        # The object payloads in this archive can deserialize torch-backed metadata,
        # which may trip duplicate libomp initialization in this local environment.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return decode_scene(npz_path, decode_auxiliary_properties=decode_auxiliary_properties)


def load_scene_properties_from_4dgs(fourdgs_path):
    """
    Decode the custom binary `.4dgs` format written from Dynamic_C3DGS `_pp.npz`.
    Supported layout:
      - xyz, motion raw float16 blocks
      - Huffman-coded scalar blocks: opacity, tcen, tsca
      - RVQ + Huffman blocks: scale, rotation, omega, tfea
      - Optional baked features_dc and rgb_dec weights
    """
    raw = _read_fourdgs_bytes(fourdgs_path)
    reader = _BinaryReader(raw)

    magic = reader.read_bytes(4)
    if magic != FOURDGS_MAGIC:
        raise ValueError(f"Unexpected 4DGS magic {magic!r} in {fourdgs_path}")

    version = reader.read_u32()
    num_points = reader.read_u32()

    xyz = reader.read_f16_array(num_points * 3).reshape(num_points, 3).astype(np.float32)
    motion = reader.read_f16_array(num_points * 9).reshape(num_points, 9).astype(np.float32)

    opacity = _read_binary_scalar_block(reader).reshape(-1, 1)
    tcen = _read_binary_scalar_block(reader)
    tsca = _read_binary_scalar_block(reader)

    scale_latent = _read_binary_vq_block(reader, num_points)
    rotation = _normalize_quaternion(_read_binary_vq_block(reader, num_points))
    omega = _read_binary_vq_block(reader, num_points)
    tfea = _read_binary_vq_block(reader, num_points)

    has_features = bool(reader.read_u8())
    features_dc = None
    if has_features:
        features_dc = reader.read_f16_array(num_points * 6).reshape(num_points, 6).astype(np.float32)

    has_rgb_dec = bool(reader.read_u8())
    rgb_decoder = None
    if has_rgb_dec:
        w1 = reader.read_f16_array(6 * 12).reshape(6, 12).astype(np.float32)
        w2 = reader.read_f16_array(3 * 6).reshape(3, 6).astype(np.float32)
        rgb_decoder = {"w1": w1, "w2": w2}

    trailing_bytes = reader.remaining()
    if trailing_bytes != 0:
        raise ValueError(f"Unexpected {trailing_bytes} trailing bytes after decoding {fourdgs_path}")

    scene = SceneProperties()
    scene.xyz = xyz
    scene.motion = motion
    scene.opacity = opacity.astype(np.float32)
    scene.tcen = tcen.astype(np.float32)
    scene.tsca = tsca.astype(np.float32)
    # In the binary produced from `_pp.npz`, scale codebooks are stored in log-space.
    scene.scale = np.exp(scale_latent).astype(np.float32)
    scene.rotation = rotation.astype(np.float32)
    scene.omega = omega.astype(np.float32)
    scene.tfea = tfea.astype(np.float32)
    scene.features_dc = features_dc
    scene.rgb_decoder = rgb_decoder
    scene.rgb = _derive_rgb_from_4dgs(scene)

    print(f"Loaded {fourdgs_path} as .4dgs version {version} with {num_points} points.")
    print(f"Loaded xyz: {scene.xyz.shape}")
    print(f"Loaded motion: {scene.motion.shape}")
    print(f"Decoded opacity: {scene.opacity.shape}")
    print(f"Decoded scale: {scene.scale.shape}")
    print(f"Decoded rotation: {scene.rotation.shape}")
    print(f"Decoded omega: {scene.omega.shape}")
    print(f"Decoded tfea: {scene.tfea.shape}")
    if scene.features_dc is not None:
        print(f"Loaded features_dc: {scene.features_dc.shape}")
    if scene.rgb_decoder is not None:
        print("Loaded rgb_dec weights")

    return scene


def load_scene_properties(input_path, decode_auxiliary_properties=False):
    if input_path.endswith(".4dgs"):
        return load_scene_properties_from_4dgs(input_path)
    if input_path.endswith(".npz"):
        return load_scene_properties_from_npz(
            input_path,
            decode_auxiliary_properties=decode_auxiliary_properties,
        )
    raise ValueError(f"Unsupported scene file type for {input_path}")


class _BinaryReader:
    def __init__(self, data):
        self.data = data
        self.offset = 0

    def read_bytes(self, count):
        if self.offset + count > len(self.data):
            raise ValueError(f"Unexpected EOF while reading {count} bytes at offset {self.offset}")
        out = self.data[self.offset:self.offset + count]
        self.offset += count
        return out

    def read_u8(self):
        return self.read_bytes(1)[0]

    def read_u16(self):
        return int(np.frombuffer(self.read_bytes(2), dtype="<u2")[0])

    def read_u32(self):
        return int(np.frombuffer(self.read_bytes(4), dtype="<u4")[0])

    def read_f32(self):
        return float(np.frombuffer(self.read_bytes(4), dtype="<f4")[0])

    def read_f16_array(self, count):
        return np.frombuffer(self.read_bytes(count * 2), dtype="<f2")

    def remaining(self):
        return len(self.data) - self.offset


def _read_fourdgs_bytes(path):
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw


def _decode_huffman_entries(entries):
    table = {}
    for symbol, bit_len, code_bits in entries:
        key = "_EOF" if symbol == 0xFFFF else int(symbol)
        table[key] = (int(bit_len), int(code_bits))
    return table


def _read_binary_huffman_block(reader):
    htable_len = reader.read_u16()
    entries = []
    for _ in range(htable_len):
        symbol = reader.read_u16()
        bit_len = reader.read_u8()
        code_bits = reader.read_u32()
        entries.append((symbol, bit_len, code_bits))

    data_len = reader.read_u32()
    payload = reader.read_bytes(data_len)
    codec = PrefixCodec(_decode_huffman_entries(entries))
    decoded = codec.decode(payload)
    return np.asarray(decoded, dtype=np.uint8)


def _read_binary_scalar_block(reader):
    min_val = reader.read_f32()
    max_val = reader.read_f32()
    quantized = _read_binary_huffman_block(reader).astype(np.float32)
    return min_val + (max_val - min_val) * (quantized / 255.0)


def _unpack_rvq_indices(bitstream, num_points, num_quantizers, rvq_bit):
    required_bits = num_points * num_quantizers * rvq_bit
    bits = np.unpackbits(np.asarray(bitstream, dtype=np.uint8), bitorder="little")[:required_bits]
    packed = np.packbits(bits.reshape(-1, rvq_bit), axis=-1, bitorder="little")
    return packed.reshape(num_points, num_quantizers).astype(np.int32)


def _read_binary_vq_block(reader, num_points):
    num_layers = reader.read_u8()
    codebook_size = reader.read_u16()
    dim = reader.read_u16()
    codebooks = reader.read_f16_array(num_layers * codebook_size * dim)
    codebooks = codebooks.astype(np.float32).reshape(num_layers, codebook_size, dim)
    packed_indices = _read_binary_huffman_block(reader)

    rvq_bit = int(round(math.log2(codebook_size)))
    if (1 << rvq_bit) != codebook_size:
        raise ValueError(f"Unsupported non-power-of-two codebook size: {codebook_size}")

    indices = _unpack_rvq_indices(packed_indices, num_points, num_layers, rvq_bit)
    return decode_rvq(indices, codebooks).astype(np.float32)


def _normalize_quaternion(rotation):
    norm = np.linalg.norm(rotation, axis=1, keepdims=True)
    norm = np.where(norm > 1e-8, norm, 1.0)
    return rotation / norm


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _derive_rgb_from_4dgs(scene):
    if scene.features_dc is None:
        return np.ones((scene.xyz.shape[0], 3), dtype=np.float32)

    # Base fallback directly from baked features.
    rgb = _sigmoid(scene.features_dc[:, :3]).astype(np.float32)

    if scene.rgb_decoder is None or scene.tfea is None or scene.tcen is None or scene.tsca is None:
        return np.clip(rgb, 0.0, 1.0)

    # Best-effort runtime color reconstruction using baked static features plus temporal feature channels.
    ones = np.ones((scene.xyz.shape[0], 1), dtype=np.float32)
    mlp_input = np.concatenate(
        [scene.features_dc, scene.tfea, scene.tcen[:, None], scene.tsca[:, None], ones],
        axis=1,
    )
    if mlp_input.shape[1] != 12:
        return np.clip(rgb, 0.0, 1.0)

    hidden = np.maximum(mlp_input @ scene.rgb_decoder["w1"].T, 0.0)
    rgb = _sigmoid(hidden @ scene.rgb_decoder["w2"].T).astype(np.float32)
    return np.clip(rgb, 0.0, 1.0)
