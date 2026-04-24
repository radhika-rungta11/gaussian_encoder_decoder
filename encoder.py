import argparse
from dataclasses import dataclass

import numpy as np

from quantize import quantize_scalar, quantize_vector
from scene_io import load_scene_properties


ENCODER_FORMAT = "gaussian_encoder_decoder_v1"


@dataclass
class EncoderOutputs:
    blobs: dict


def _normalize_quaternion(rotation):
    rotation = np.asarray(rotation, dtype=np.float32)
    norm = np.linalg.norm(rotation, axis=1, keepdims=True)
    norm = np.where(norm > 1e-8, norm, 1.0)
    return rotation / norm


def encode_scene_properties(scene_properties):
    """
    Encode scene properties into a compact NPZ format that our decoder can load directly.
    This codec is intentionally simple and robust:
      - geometry and motion: uint16 min-max quantization
      - opacity / RGB: uint8 min-max quantization
      - scale: uint16 min-max quantization
      - rotation: float16 normalized quaternion
    """
    blobs = {
        "codec_format": np.array(ENCODER_FORMAT),
    }

    if scene_properties.xyz is None:
        raise ValueError("Cannot encode a scene without xyz positions.")

    xyz_q, xyz_min, xyz_max = quantize_vector(scene_properties.xyz, num_bits=16)
    blobs["xyz_q"] = xyz_q
    blobs["xyz_min"] = xyz_min
    blobs["xyz_max"] = xyz_max

    if scene_properties.motion is not None:
        motion_q, motion_min, motion_max = quantize_vector(scene_properties.motion, num_bits=16)
        blobs["motion_q"] = motion_q
        blobs["motion_min"] = motion_min
        blobs["motion_max"] = motion_max

    if scene_properties.opacity is not None:
        opacity_q, opacity_minmax = quantize_scalar(scene_properties.opacity.reshape(-1), num_bits=8)
        blobs["opacity_q"] = opacity_q
        blobs["opacity_minmax"] = opacity_minmax

    if scene_properties.scale is not None:
        scale_q, scale_min, scale_max = quantize_vector(scene_properties.scale, num_bits=16)
        blobs["scale_q"] = scale_q
        blobs["scale_min"] = scale_min
        blobs["scale_max"] = scale_max

    if scene_properties.rotation is not None:
        blobs["rotation_f16"] = _normalize_quaternion(scene_properties.rotation).astype(np.float16)

    if scene_properties.rgb is not None:
        rgb_q, rgb_min, rgb_max = quantize_vector(scene_properties.rgb, num_bits=8)
        blobs["rgb_q"] = rgb_q
        blobs["rgb_min"] = rgb_min
        blobs["rgb_max"] = rgb_max

    for name in ["tcen", "tsca", "omega", "tfea"]:
        value = getattr(scene_properties, name, None)
        if value is None:
            continue
        if value.ndim == 1:
            value_q, value_minmax = quantize_scalar(value, num_bits=16)
            blobs[f"{name}_q"] = value_q
            blobs[f"{name}_minmax"] = value_minmax
        else:
            value_q, value_min, value_max = quantize_vector(value, num_bits=16)
            blobs[f"{name}_q"] = value_q
            blobs[f"{name}_min"] = value_min
            blobs[f"{name}_max"] = value_max

    return EncoderOutputs(blobs=blobs)


def save_encoded_scene(scene_properties, output_path):
    encoded = encode_scene_properties(scene_properties)
    np.savez_compressed(output_path, **encoded.blobs)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Encode a Gaussian scene into a compact NPZ archive.")
    parser.add_argument("input_path", help="Input .npz or .4dgs scene file")
    parser.add_argument("output_path", help="Output encoded .npz file")
    parser.add_argument(
        "--decode-aux",
        action="store_true",
        help="When the input is the original compressed NPZ, decode auxiliary Gaussian properties before re-encoding.",
    )
    args = parser.parse_args()

    scene = load_scene_properties(args.input_path, decode_auxiliary_properties=args.decode_aux)
    save_encoded_scene(scene, args.output_path)
    print(f"Saved encoded scene to {args.output_path}")


if __name__ == "__main__":
    main()
