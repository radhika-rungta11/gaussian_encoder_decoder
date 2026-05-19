import numpy as np
from quantize import inverse_quantize_scalar, inverse_quantize_vector, decode_rvq

from dahuffman.huffmancodec import PrefixCodec


def _extract_rgb_array(rgb_payload, num_points):
    """
    Best-effort conversion of an NPZ RGB payload into an (N, 3) float array in [0, 1].

    Some archives store `rgb_dec` as a serialized model/state-dict rather than a decoded
    per-point color tensor. In that case we return None so the caller can use a safe fallback.
    """
    if rgb_payload is None:
        return None

    if isinstance(rgb_payload, np.ndarray) and rgb_payload.shape == ():
        rgb_payload = rgb_payload.item()

    if isinstance(rgb_payload, dict):
        return None

    rgb = np.asarray(rgb_payload)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        return None

    rgb = rgb.astype(np.float32, copy=False)
    if rgb.shape[0] != num_points:
        return None

    # Normalize common uint8 color data; otherwise assume values are already floats.
    if np.issubdtype(rgb.dtype, np.integer) or np.max(rgb) > 1.0:
        rgb = rgb / 255.0

    return np.clip(rgb, 0.0, 1.0)


def _decode_rvq_feature(data, value_key, table_key, codebook_key, num_points, num_quantizers, rvq_bit):
    symbols = _decode_huffman_payload(data, value_key, table_key)
    indices = _decode_rvq_indices(symbols, num_points, int(num_quantizers), int(rvq_bit))
    codebook = _extract_rvq_codebook(data[codebook_key])
    return decode_rvq(indices, codebook).astype(np.float32)


def _decode_rvq_indices(symbols, num_points, num_quantizers, rvq_bit):
    """
    Two on-disk layouts exist in the wild:
      1. Huffman alphabet IS the RVQ index alphabet (one symbol per index).
      2. Huffman alphabet is bytes (0-255), and the byte stream is the bit-packed
         indices that need np.unpackbits to recover.
    Detect which one by checking whether the symbol count matches the expected
    index count.
    """
    expected = num_points * num_quantizers
    max_index_value = (1 << int(rvq_bit)) - 1
    if symbols.size == expected and int(symbols.max(initial=0)) <= max_index_value:
        return symbols.reshape(num_points, num_quantizers).astype(np.int32)
    byte_stream = symbols.astype(np.uint8)
    return _unpack_rvq_indices(byte_stream, num_points, int(num_quantizers), int(rvq_bit))


def _decode_quantized_scalar_feature(data, value_key, table_key, minmax_key):
    quantized = _decode_huffman_payload(data, value_key, table_key)
    min_val, max_val = data[minmax_key].astype(np.float32)
    return inverse_quantize_scalar(quantized, min_val, max_val).astype(np.float32)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _derive_rgb_from_feature_network(scene, data):
    """
    Best-effort color reconstruction from the saved appearance network weights.

    Inference from the archive structure:
    - `rgb_dec` stores a tiny 2-layer 1x1 Conv MLP with shapes 12->6->3.
    - The likely input bundle is a concatenation of normalized xyz, temporal features,
      omega coefficients, and scalar temporal controls.
    """
    if 'rgb_dec' not in data:
        return None

    payload = data['rgb_dec']
    if isinstance(payload, np.ndarray) and payload.shape == ():
        payload = payload.item()

    if not isinstance(payload, dict):
        return None

    required_scene_attrs = (scene.xyz, scene.tfea, scene.omega, scene.tcen, scene.tsca)
    if any(attr is None for attr in required_scene_attrs):
        return None

    if 'mlp1.weight' not in payload or 'mlp2.weight' not in payload:
        return None

    w1 = payload['mlp1.weight']
    w2 = payload['mlp2.weight']
    w1 = w1.detach().cpu().numpy() if hasattr(w1, "detach") else np.asarray(w1)
    w2 = w2.detach().cpu().numpy() if hasattr(w2, "detach") else np.asarray(w2)
    w1 = np.asarray(w1, dtype=np.float32).reshape(6, 12)
    w2 = np.asarray(w2, dtype=np.float32).reshape(3, 6)

    xyz_min = np.min(scene.xyz, axis=0, keepdims=True)
    xyz_max = np.max(scene.xyz, axis=0, keepdims=True)
    xyz_span = np.maximum(xyz_max - xyz_min, 1e-5)
    xyz_normalized = (scene.xyz - xyz_min) / xyz_span

    mlp_input = np.concatenate(
        [
            xyz_normalized,
            scene.tfea,
            scene.omega,
            scene.tcen.reshape(-1, 1),
            scene.tsca.reshape(-1, 1),
        ],
        axis=1,
    )
    hidden = np.maximum(mlp_input @ w1.T, 0.0)
    rgb = _sigmoid(hidden @ w2.T).astype(np.float32)
    return np.clip(rgb, 0.0, 1.0)


def _decode_huffman_payload(data, value_key, table_key):
    """
    Decode a Huffman-coded payload stored in the NPZ. Returns int32 because
    the alphabet can be larger than 256 entries (RVQ indices for codebooks
    with rvq_bit > 8).
    """
    codec = PrefixCodec(data[table_key].item())
    decoded = codec.decode(data[value_key])
    return np.asarray(decoded, dtype=np.int32)


def _extract_rvq_codebook(codebook_payload):
    """
    Convert the saved ResidualVQ state dict into a stacked numpy codebook array.
    """
    if isinstance(codebook_payload, np.ndarray) and codebook_payload.shape == ():
        codebook_payload = codebook_payload.item()

    if not isinstance(codebook_payload, dict):
        raise TypeError("Expected RVQ codebook payload to be a dict-like state dict.")

    stage_entries = []
    for key, value in codebook_payload.items():
        if not key.endswith("._codebook.embed"):
            continue

        stage_idx = int(key.split(".")[1])
        arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        stage_entries.append((stage_idx, arr))

    if not stage_entries:
        raise ValueError("No RVQ embed tensors were found in the codebook payload.")

    stage_entries.sort(key=lambda item: item[0])
    return np.stack([entry[1] for entry in stage_entries], axis=0)


def _unpack_rvq_indices(bitstream, num_points, num_quantizers, rvq_bit):
    """
    Recover RVQ indices from the bit-packed byte stream saved in the NPZ.
    Handles any rvq_bit (including codebooks larger than 256 entries, which
    need rvq_bit > 8 and therefore can't fit a single uint8 byte).
    """
    required_bits = num_points * num_quantizers * rvq_bit
    bits = np.unpackbits(np.asarray(bitstream, dtype=np.uint8), bitorder='little')
    bits = bits[:required_bits]

    if bits.size != required_bits:
        raise ValueError(
            f"RVQ bitstream is too short: expected {required_bits} bits, got {bits.size}."
        )

    bit_matrix = bits.reshape(-1, rvq_bit).astype(np.int32)
    weights = (1 << np.arange(rvq_bit, dtype=np.int32))
    indices = (bit_matrix * weights[None, :]).sum(axis=1)
    return indices.reshape(num_points, num_quantizers)


def _normalize_quaternion(rotation):
    norm = np.linalg.norm(rotation, axis=1, keepdims=True)
    norm = np.where(norm > 1e-8, norm, 1.0)
    return rotation / norm


class SceneProperties:
    """
    Container for the decoded scene properties.
    """
    def __init__(self):
        self.xyz = None
        self.motion = None
        self.opacity = None
        self.scale = None
        self.rotation = None
        self.rgb = None
        self.tcen = None
        self.tsca = None
        self.omega = None
        self.tfea = None
        self.features_dc = None
        self.rgb_decoder = None
        self.sh_features = None


def _decode_simple_encoded_scene(data):
    scene = SceneProperties()

    scene.xyz = inverse_quantize_vector(data["xyz_q"], data["xyz_min"], data["xyz_max"], num_bits=16).astype(np.float32)

    if {"motion_q", "motion_min", "motion_max"}.issubset(data.files):
        scene.motion = inverse_quantize_vector(
            data["motion_q"], data["motion_min"], data["motion_max"], num_bits=16
        ).astype(np.float32)

    num_points = scene.xyz.shape[0]

    scene.opacity = np.ones((num_points, 1), dtype=np.float32)
    if {"opacity_q", "opacity_minmax"}.issubset(data.files):
        scene.opacity = inverse_quantize_scalar(
            data["opacity_q"], data["opacity_minmax"][0], data["opacity_minmax"][1], num_bits=8
        ).reshape(-1, 1).astype(np.float32)

    scene.scale = np.ones((num_points, 3), dtype=np.float32) * 0.01
    if {"scale_q", "scale_min", "scale_max"}.issubset(data.files):
        scene.scale = inverse_quantize_vector(
            data["scale_q"], data["scale_min"], data["scale_max"], num_bits=16
        ).astype(np.float32)

    scene.rotation = np.zeros((num_points, 4), dtype=np.float32)
    scene.rotation[:, 0] = 1.0
    if "rotation_f16" in data.files:
        scene.rotation = _normalize_quaternion(data["rotation_f16"].astype(np.float32))

    scene.rgb = np.ones((num_points, 3), dtype=np.float32)
    if {"rgb_q", "rgb_min", "rgb_max"}.issubset(data.files):
        scene.rgb = inverse_quantize_vector(data["rgb_q"], data["rgb_min"], data["rgb_max"], num_bits=8).astype(
            np.float32
        )

    for name in ["tcen", "tsca", "omega", "tfea"]:
        if f"{name}_q" not in data.files:
            continue
        if f"{name}_minmax" in data.files:
            setattr(
                scene,
                name,
                inverse_quantize_scalar(
                    data[f"{name}_q"],
                    data[f"{name}_minmax"][0],
                    data[f"{name}_minmax"][1],
                    num_bits=16,
                ).astype(np.float32),
            )
        elif {f"{name}_min", f"{name}_max"}.issubset(data.files):
            setattr(
                scene,
                name,
                inverse_quantize_vector(
                    data[f"{name}_q"], data[f"{name}_min"], data[f"{name}_max"], num_bits=16
                ).astype(np.float32),
            )

    print(f"Loaded encoded scene format with xyz: {scene.xyz.shape}")
    return scene


def decode_scene(npz_file_path, decode_auxiliary_properties=False):
    """
    The Decoder Input is the `.npz` file provided by the encoder.
    This function reads the compressed arrays, unpacks them using minmax ranges,
    huffman tables, or codebooks, and returns raw float arrays suitable for rendering.
    """
    print(f"Decoding {npz_file_path}...")
    data = np.load(npz_file_path, allow_pickle=True)

    if "codec_format" in data.files:
        codec_format = str(np.asarray(data["codec_format"]).item())
        if codec_format == "gaussian_encoder_decoder_v1":
            return _decode_simple_encoded_scene(data)
        raise ValueError(f"Unsupported encoded scene format: {codec_format}")

    scene = SceneProperties()

    # Base geometry and motion are already float16.
    if 'xyz' in data:
        scene.xyz = data['xyz'].astype(np.float32)
        print(f"Loaded xyz: {scene.xyz.shape}")

    if 'motion' in data:
        scene.motion = data['motion'].astype(np.float32)
        print(f"Loaded motion: {scene.motion.shape}")

    num_points = scene.xyz.shape[0] if scene.xyz is not None else 0

    scene.opacity = np.ones((num_points, 1), dtype=np.float32)
    if {'opacity', 'huftable_opacity', 'minmax_opacity'}.issubset(data.files):
        quantized_opacity = _decode_huffman_payload(data, 'opacity', 'huftable_opacity')
        min_opacity, max_opacity = data['minmax_opacity'].astype(np.float32)
        decoded_opacity = inverse_quantize_scalar(quantized_opacity, min_opacity, max_opacity)
        scene.opacity = decoded_opacity.reshape(-1, 1).astype(np.float32)
        print(f"Decoded opacity: {scene.opacity.shape}")

    scene.scale = np.ones((num_points, 3), dtype=np.float32) * 0.01
    scene.rotation = np.zeros((num_points, 4), dtype=np.float32)
    scene.rotation[:, 0] = 1.0

    rvq_info_key = 'rvq_info_geo' if 'rvq_info_geo' in data else 'rvq_info'
    if decode_auxiliary_properties and rvq_info_key in data:
        rvq_num, rvq_bit = data[rvq_info_key].astype(np.int32)

        if {'scale', 'huftable_scale', 'codebook_scale'}.issubset(data.files):
            scale_symbols = _decode_huffman_payload(data, 'scale', 'huftable_scale')
            scale_indices = _decode_rvq_indices(scale_symbols, num_points, int(rvq_num), int(rvq_bit))
            scale_codebook = _extract_rvq_codebook(data['codebook_scale'])
            scene.scale = decode_rvq(scale_indices, scale_codebook).astype(np.float32)
            scene.scale = np.clip(scene.scale, 1e-6, None)
            print(f"Decoded scale: {scene.scale.shape}")

        if {'rotation', 'huftable_rotation', 'codebook_rotation'}.issubset(data.files):
            rotation_symbols = _decode_huffman_payload(data, 'rotation', 'huftable_rotation')
            rotation_indices = _decode_rvq_indices(rotation_symbols, num_points, int(rvq_num), int(rvq_bit))
            rotation_codebook = _extract_rvq_codebook(data['codebook_rotation'])
            scene.rotation = decode_rvq(rotation_indices, rotation_codebook).astype(np.float32)
            scene.rotation = _normalize_quaternion(scene.rotation)
            print(f"Decoded rotation: {scene.rotation.shape}")

    rvq_temp_key = 'rvq_info_temp'
    if decode_auxiliary_properties and rvq_temp_key in data:
        rvq_num_temp, rvq_bit_temp = data[rvq_temp_key].astype(np.int32)

        if {'omega', 'huftable_omega', 'codebook_omega'}.issubset(data.files):
            scene.omega = _decode_rvq_feature(
                data,
                'omega',
                'huftable_omega',
                'codebook_omega',
                num_points,
                rvq_num_temp,
                rvq_bit_temp,
            )
            print(f"Decoded omega: {scene.omega.shape}")

        if {'tfea', 'huftable_tfea', 'codebook_tfea'}.issubset(data.files):
            scene.tfea = _decode_rvq_feature(
                data,
                'tfea',
                'huftable_tfea',
                'codebook_tfea',
                num_points,
                rvq_num_temp,
                rvq_bit_temp,
            )
            print(f"Decoded tfea: {scene.tfea.shape}")

        if {'tcen', 'huftable_tcen', 'minmax_tcen'}.issubset(data.files):
            scene.tcen = _decode_quantized_scalar_feature(
                data,
                'tcen',
                'huftable_tcen',
                'minmax_tcen',
            )
            print(f"Decoded tcen: {scene.tcen.shape}")

        if {'tsca', 'huftable_tsca', 'minmax_tsca'}.issubset(data.files):
            scene.tsca = _decode_quantized_scalar_feature(
                data,
                'tsca',
                'huftable_tsca',
                'minmax_tsca',
            )
            print(f"Decoded tsca: {scene.tsca.shape}")
    elif rvq_info_key in data:
        print("Skipping auxiliary Gaussian property decode (scale/rotation/RGB metadata) for safe point-cloud rendering.")

    # Mock Color / RGB.
    scene.rgb = np.ones((num_points, 3), dtype=np.float32)
    if decode_auxiliary_properties and 'rgb_dec' in data:
        decoded_rgb = _extract_rgb_array(data['rgb_dec'], num_points)
        if decoded_rgb is not None:
            scene.rgb = decoded_rgb
            print(f"Loaded rgb: {scene.rgb.shape}")
        else:
            derived_rgb = _derive_rgb_from_feature_network(scene, data)
            if derived_rgb is not None:
                scene.rgb = derived_rgb
                print(f"Derived rgb from feature network: {scene.rgb.shape}")
            else:
                print("Warning: 'rgb_dec' contains model metadata, not a per-point RGB array. Using white fallback colors.")
    elif 'rgb_dec' in data:
        print("Skipping 'rgb_dec' object payload and using fallback colors for safe rendering.")

    return scene


if __name__ == "__main__":
    import sys
    path = "point_cloud_pp.npz"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    scene = decode_scene(path)
    print("Decoding complete. Points:", len(scene.xyz))
