"""Frozen FP32 encoder extraction; GPU commands are for Colab, never the laptop.

describe prints the fixed recipes without importing a framework. spec hashes staged
assets after terms/access approval. pilot runs the fixed, score-free timing gate;
extract needs the archived input lock and, with --external, the external query lock.
Every process loads exactly one encoder.
See docs/research/FM_RUNBOOK.md for the spec and gate schemas and commands.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata, util
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np
from PIL import Image

import fm_core as fm
import fm_prepare as prep

REPOSITORIES = {
    "RETFound": "YukunZhou/RETFound_mae_natureOCT",
    "DINOv2": "facebook/dinov2-base",
    "MIRAGE": "j-morano/MIRAGE-Base",
    "BioMedCLIP": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "Inception": "keras.applications.InceptionV3/imagenet",
}
DIMENSIONS = dict(RETFound=1024, DINOv2=768, MIRAGE=768, BioMedCLIP=512, Inception=2048)
REQUIRED_ASSETS = {
    "RETFound": ("RETFound_mae_natureOCT.pth", "models_vit.py"),
    "DINOv2": ("model.safetensors", "config.json", "preprocessor_config.json"),
    "MIRAGE": ("model.safetensors", "config.json", "mirage_hf.py"),
    "BioMedCLIP": ("open_clip_pytorch_model.bin", "open_clip_config.json"),
    "Inception": ("inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",),
}
SOURCE_REPOS = {"RETFound": "https://github.com/rmaphoh/RETFound_MAE",
                "MIRAGE": "https://github.com/j-morano/MIRAGE"}
REVISIONS = {
    "RETFound": "8e551e2f547d3e27aedf43c9cf1c5bcea1c17171",
    "DINOv2": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
    "MIRAGE": "473aadadae9940fcbeccd197db1bc3fac6b376fb",
    "BioMedCLIP": "9f341de24bfb00180f1b847274256e9b65a3a32e",
    "Inception": "keras-imagenet-notop",
}
SOURCE_REVISIONS = {"RETFound": "ae9a9ecf37857cf47b8aa9f87cd6f710d75db287",
                    "MIRAGE": "ee5eae8d3524ec791e719ced1f3cca27eae4c1b0"}
IMAGENET_MEAN = [.485, .456, .406]
IMAGENET_STD = [.229, .224, .225]
# HF LFS metadata and exact revision-pinned config/source bytes, checked 9 Nov.
# Inception SHA256: existing historical cache, whose MD5 matches Keras' official
# no-top hash bcbd6486424b2319ff4ef7d526e38f63. No checkpoint download was needed.
EXPECTED_ASSET_SHA256 = {
    "RETFound": {
        "RETFound_mae_natureOCT.pth":"e9ff7864f40334885062953cb3894739d9fff52aaeb22c8b4f47d92270068d18",
        "models_vit.py":"f5a91312fcbac14141405d24a92ac3a95e2573e98260bbd3c5b044792edd2f75"},
    "DINOv2": {
        "model.safetensors":"d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841",
        "config.json":"f7ff4cfa73d2f70647dbf6950541ad25d73082d54c2e7e9bded160c7656b2a70",
        "preprocessor_config.json":"14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828"},
    "MIRAGE": {
        "model.safetensors":"bdb3ad174fd8106f32e692446479966ebf095fa8f9a523f9b29185ed0136c339",
        "config.json":"55ece6b8cac31a879543d13e8e2ae95a0174945a198b0187176dc337488ddf28",
        "mirage_hf.py":"07fa63e2766b67dbc0c07f0d9d29db1562f707425ef2fbf90a688f62a94e6ede"},
    "BioMedCLIP": {
        "open_clip_pytorch_model.bin":"52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be",
        "open_clip_config.json":"9a41f334a8c444678772c0ebb9ab854c97ab350bced3a17b803e258d39c23dc0"},
    "Inception": {
        "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5":"9617109a16463f250180008f9818336b767bdf5164315e8cd5761a8c34caa62a"},
}


def recipe(arm: str) -> dict:
    """One prespecified embedding per arm; no outcome-selected pooling options."""
    recipes = {
        "RETFound": dict(channels="RGB (alpha dropped)", resize="short edge 256 bicubic", crop=224,
                         normalization="ImageNet mean/std", pooling="patch-token mean then fc_norm",
                         source="models_vit.py:VisionTransformer.forward_features; util/datasets.py eval"),
        "DINOv2": dict(channels="RGB (alpha dropped)", resize="preprocessor_config.json", crop=224,
                       normalization="preprocessor_config.json", pooling="pooler_output (normalized CLS)",
                       source="facebook/dinov2-base preprocessor_config.json; transformers Dinov2Model"),
        # A2 A-11: the published frozen-probe input path (fm_cls_config.py MIRAGEFM), not the HF demo.
        "MIRAGE": dict(channels="native PIL resize, then L (alpha dropped)", resize="512x512 bilinear", crop=None,
                       normalization="divide by 255, then per-image min-max to [0, 1]",
                       pooling="LayerNorm(eps=1e-6) per token, patch-token mean; global excluded",
                       source="fm_cls_config.py:MIRAGEFM (MinMaxNormChannel); mirage_wrapper.py:MIRAGEClsGlobal.forward/pool"),
        "BioMedCLIP": dict(channels="native eval RGB conversion after resize/crop (alpha dropped)", resize="open_clip native eval transform", crop=224,
                           normalization="open_clip_config.json", pooling="projected image feature, normalize=False",
                           source="open_clip_config.json; open_clip image_transform_v2; model.visual"),
        "Inception": dict(channels="stage3_classifier.load_image channels=3", resize="bilinear 256x256 then299x299",
                          crop=None, normalization="pixels / 127.5 - 1", pooling="Keras global average pooling",
                          source="stage3_classifier.load_image/backbone; historical Keras ImageNet weights"),
    }
    return dict(arm=arm, dimension=DIMENSIONS[arm], dtype="float32", inference_mode=True,
                batch_size=1, **recipes[arm])


def environment() -> dict:
    """Record the actual environment, including transitive dependencies, without secrets."""
    packages = {dist.metadata["Name"].lower(): dist.version for dist in metadata.distributions()
                if dist.metadata["Name"]}
    return dict(python=platform.python_version(), platform=platform.platform(), packages=dict(sorted(packages.items())))


def hardware() -> dict[str, str]:
    """Same host/boot and GPU identity for sequential timing, not just a shared GPU name."""
    boot = Path("/proc/sys/kernel/random/boot_id")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                                   "--format=csv,noheader"], text=True).strip()
    return dict(host=platform.node(), boot_id=boot.read_text().strip() if boot.exists() else "unverified", gpu=gpu)


def require_colab() -> None:
    if platform.system() != "Linux" or not Path("/content").is_dir():
        raise ValueError("pilot/extract run only in the Colab runtime, not on the laptop")


def runtime_out(out: Path) -> Path:
    out = out.resolve()
    if not out.is_relative_to(Path("/content/fm_v1")) or "drive" in out.parts:
        raise ValueError("runtime outputs must be under /content/fm_v1, outside Drive")
    return out


def make_spec(arm: str, assets: Path, revision: str, source_revision: str, terms: str) -> dict:
    """Hash already acquired assets; this command never downloads or accepts terms."""
    if not terms.strip() or (arm != "Inception" and not re.fullmatch(r"[0-9a-f]{40}", revision)):
        raise ValueError("terms evidence and full immutable checkpoint revision are required")
    if arm in SOURCE_REPOS and not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("RETFound/MIRAGE also need the full upstream code revision")
    spec = dict(arm=arm, repository=REPOSITORIES[arm], revision=revision,
                source_revision=source_revision, terms_evidence=terms,
                files={name: prep.sha256(assets / name) for name in REQUIRED_ASSETS[arm]}, recipe=recipe(arm))
    check_spec(spec)
    return spec


def check_spec(spec: dict, assets: Path | None = None) -> None:
    arm = spec["arm"]
    if arm not in REPOSITORIES or spec["repository"] != REPOSITORIES[arm] or spec["recipe"] != recipe(arm):
        raise ValueError("checkpoint or recipe is outside the fixed roster")
    if set(spec["files"]) != set(REQUIRED_ASSETS[arm]) or not spec["terms_evidence"].strip():
        raise ValueError("incomplete model identity or terms evidence")
    if arm != "Inception" and not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]):
        raise ValueError("checkpoint revision must be immutable")
    if arm in SOURCE_REPOS and not re.fullmatch(r"[0-9a-f]{40}", spec["source_revision"]):
        raise ValueError("upstream source revision must be immutable")
    if arm in REVISIONS and spec["revision"] != REVISIONS[arm]:
        raise ValueError("checkpoint revision differs from the prespecified source pin")
    if arm in SOURCE_REVISIONS and spec["source_revision"] != SOURCE_REVISIONS[arm]:
        raise ValueError("upstream source revision differs from the prespecified source pin")
    if not spec["revision"].strip() or any(not re.fullmatch(r"[0-9a-f]{64}", digest)
                                           for digest in spec["files"].values()):
        raise ValueError("checkpoint revision and SHA256 asset hashes are required")
    if spec["files"] != EXPECTED_ASSET_SHA256[arm]:
        raise ValueError("staged weight/config/source hashes differ from the verified public source pins")
    if assets is not None:
        prep.verify_files(assets, spec["files"])


def prepared(folder: Path) -> dict:
    return prep.verify_preparation(folder)


def validate_features(values: np.ndarray, count: int, dimension: int) -> None:
    if count < 1 or dimension < 1 or values.dtype != np.float32 or values.shape != (count, dimension):
        raise ValueError(f"expected raw float32 embeddings of shape {(count, dimension)}; got {values.shape}/{values.dtype}")
    fm.unit(values)  # also rejects nonfinite and zero vectors


def validate_inventory(paths: Sequence[str], hashes: Sequence[str]) -> None:
    if not paths or len(set(paths)) != len(paths) or len(hashes) != len(paths):
        raise ValueError("feature IDs/hashes must be nonempty, unique and aligned")
    if any(prep.safe_id(path) != path for path in paths) or any(
            not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes):
        raise ValueError("feature IDs must be canonical and content hashes must be SHA256")


def save_features(path: Path, paths: Sequence[str], hashes: Sequence[str], values: np.ndarray) -> float:
    """Validate raw arrays, atomically save, and return serialization-only seconds."""
    validate_inventory(paths, hashes)
    if values.ndim != 2:
        raise ValueError("features must be a two-dimensional float32 array")
    validate_features(values, len(paths), values.shape[1])
    start = time.perf_counter()
    buffer = BytesIO()
    np.savez(buffer, paths=np.array(paths), sha256=np.array(hashes), feats=values)
    prep.write_once(path, buffer.getvalue())
    return time.perf_counter() - start


def load_features(path: Path, inventory: prep.Rows, dimension: int) -> np.ndarray:
    paths, hashes = [r["path"] for r in inventory], [r["sha256"] for r in inventory]
    validate_inventory(paths, hashes)
    with np.load(path, allow_pickle=False) as cache:
        if sorted(cache.files) != ["feats", "paths", "sha256"]:
            raise ValueError("feature cache must contain exactly paths, sha256 and feats")
        ids, digests = cache["paths"], cache["sha256"]
        if ids.dtype.kind != "U" or digests.dtype.kind != "U" or ids.ndim != 1 or digests.ndim != 1:
            raise ValueError("feature IDs/hashes must be one-dimensional Unicode arrays")
        if ids.tolist() != paths or digests.tolist() != hashes:
            raise ValueError("feature paths/content hashes differ from the ordered inventory")
        values = cache["feats"]
    validate_features(values, len(inventory), dimension)
    return values


def mirage_input(image: Image.Image) -> np.ndarray:
    """MIRAGE frozen-probe input (A2 A-11): 512x512 bilinear, L, /255, per-image min-max.

    Matches fm_cls_config.py MIRAGEFM at the pinned commit; a constant image stops the arm
    instead of dividing by zero.
    """
    pixels = np.array(image.resize((512, 512), Image.Resampling.BILINEAR).convert("L"), dtype=np.float32) / 255
    low, high = pixels.min(), pixels.max()
    if not high > low:
        raise ValueError("MIRAGE min-max input needs a nonconstant image")
    return (pixels - low) / (high - low)


def geometry(arm: str, path: str, image: Image.Image) -> list:
    """Log resize/crop geometry, retained original field and the A2 padding shares.

    Full-field warps retain 1. Padding is measured on the decoded 8-bit grayscale original.
    """
    if arm not in REPOSITORIES:
        raise ValueError("unknown encoder geometry")
    gray = np.asarray(image.convert("RGB").convert("L"))
    w, h = image.size
    if arm in ("MIRAGE", "Inception"):
        rw, rh = (512, 512) if arm == "MIRAGE" else (299, 299)
        cw, ch, left, top = rw, rh, 0, 0
    else:
        # RETFound/DINOv2 short edge 256; open_clip default eval short edge 224.
        short = 224 if arm == "BioMedCLIP" else 256
        rw, rh = (short, int(short * h / w)) if w <= h else (int(short * w / h), short)
        cw, ch = min(224, rw), min(224, rh)
        # HF's slow BitImageProcessor floors; torchvision/OpenCLIP use round-to-even.
        if arm == "DINOv2":
            left, top = (rw - 224) // 2, (rh - 224) // 2
        else:
            left, top = int(round((rw - 224) / 2)), int(round((rh - 224) / 2))
    retained = (cw * ch) / (rw * rh)
    return [path, w, h, rw, rh, left, top, cw, ch, retained, fm.border_share(gray), fm.near_white_share(gray)]


@dataclass
class Encoder:
    arm: str
    encode: Callable[[Sequence[Image.Image]], np.ndarray]
    synchronize: Callable[[], None]
    load_report: dict
    peak_memory: Callable[[], dict]
    # Keras stages via stage3.load_image, then returns an inference-only callable.
    stage_path: Callable[[Path], Callable[[], np.ndarray]] | None = None
    # The reference pilot uses TF's decoder too; PIL JPEG decoding can differ.
    decode_path: Callable[[Path], Image.Image] | None = None


def import_source(path: Path):
    module_spec = util.spec_from_file_location("fm_upstream_encoder", path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError("cannot load pinned upstream source")
    module = util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def load_retfound_checkpoint(path: Path) -> dict:
    """Restricted CPU metadata load, without constructing an encoder.

    Production callers first verify the asset hash. The authors' writer includes
    argparse metadata; only that container is temporarily added to the allowlist.
    """
    import torch
    added = [] if argparse.Namespace in torch.serialization.get_safe_globals() else [argparse.Namespace]
    with torch.serialization.safe_globals(added):
        checkpoint = torch.load(path,map_location="cpu",weights_only=True)
    if not isinstance(checkpoint,dict):
        raise ValueError("RETFound checkpoint must be a state/metadata mapping")
    return checkpoint


def load_encoder(spec: dict, assets: Path) -> Encoder:
    """Use only local, identity-checked assets. No automatic checkpoint search/download."""
    require_colab()
    check_spec(spec, assets)
    arm = spec["arm"]
    if arm == "Inception":
        if "torch" in sys.modules:
            raise ValueError("the Keras reference needs a fresh process without PyTorch")
        import stage3_classifier as s3
        tf = s3.need_tf()
        if not tf.config.list_physical_devices("GPU"):
            raise ValueError("Colab CUDA GPU is required for matched extraction/timing")
        tf.keras.mixed_precision.set_global_policy("float32")
        tf.config.experimental.enable_tensor_float_32_execution(False)
        tf.config.experimental.reset_memory_stats("GPU:0")
        model = s3.backbone(weights="None")
        model.load_weights(str(assets / REQUIRED_ASSETS[arm][0]))
        model.trainable = False

        def keras_batch(images):
            # Used only for warm timing, with decoding excluded but both historical resizes included.
            tensors = [tf.image.resize(tf.image.resize(np.asarray(im.convert("RGB")), (256, 256)), (299, 299))
                       / 127.5 - 1. for im in images]
            return model(tf.stack(tensors), training=False).numpy()

        def keras_path(path):
            inputs = s3.load_image(str(path))[None]
            return lambda: model(inputs, training=False).numpy()

        def keras_decode(path):
            pixels = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
            return Image.fromarray(pixels.numpy())

        def keras_memory():
            return dict(peak_vram_bytes=tf.config.experimental.get_memory_info("GPU:0")["peak"])

        return Encoder(arm, keras_batch, tf.test.experimental.sync_devices,
                       dict(weights="strict Keras load", dtype="float32"), keras_memory, keras_path, keras_decode)

    # Frameworks stay isolated; imports here never occur on a CPU evaluator/help path.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["USE_TF"] = "0"  # transformers.image_transforms otherwise imports installed TensorFlow.
    os.environ["USE_FLAX"] = "0"
    if "tensorflow" in sys.modules:
        raise ValueError("PyTorch extraction needs a fresh process without TensorFlow")
    import torch
    from torchvision import transforms
    if not torch.cuda.is_available():
        raise ValueError("Colab CUDA GPU is required for matched extraction/timing")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    report = dict(missing_keys=[], unexpected_keys=[])

    if arm == "DINOv2":
        from transformers import AutoImageProcessor, Dinov2Model
        config = prep.read_json(assets / "preprocessor_config.json")
        expected = dict(crop_size=dict(height=224, width=224), do_center_crop=True, do_convert_rgb=True,
                        do_normalize=True, do_rescale=True, do_resize=True, image_mean=IMAGENET_MEAN,
                        image_std=IMAGENET_STD, image_processor_type="BitImageProcessor", resample=3,
                        rescale_factor=1 / 255, size=dict(shortest_edge=256))
        if config != expected:
            raise ValueError("DINOv2 preprocessing differs from the pinned native recipe/geometry")
        processor = AutoImageProcessor.from_pretrained(str(assets), local_files_only=True, use_fast=False)
        model, info = Dinov2Model.from_pretrained(str(assets), local_files_only=True, use_safetensors=True,
                                                 output_loading_info=True, attn_implementation="eager")
        if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"DINOv2 loading mismatch: {info}")
        report.update(info)
        prepare = lambda images: processor(images=list(images), return_tensors="pt")["pixel_values"]
        forward = lambda x: model(pixel_values=x).pooler_output
    elif arm == "RETFound":
        source = import_source(assets / "models_vit.py")
        model = source.RETFound_mae(global_pool=True)
        checkpoint = load_retfound_checkpoint(assets / REQUIRED_ASSETS[arm][0])
        report["checkpoint_metadata_allowlist"] = ["argparse.Namespace"]
        state = checkpoint["model"]
        state = {k: v for k, v in state.items() if k not in ("head.weight", "head.bias")}
        info = model.load_state_dict(state, strict=False)
        missing, unexpected = list(info.missing_keys), list(info.unexpected_keys)
        head = {"head.weight", "head.bias"}
        norm = {"fc_norm.weight", "fc_norm.bias"}
        if set(missing) not in (head, head | norm) or any(not (k.startswith("decoder_") or k in
                ("mask_token", "norm.weight", "norm.bias")) for k in unexpected):
            raise ValueError(f"RETFound loading mismatch: missing={missing}, unexpected={unexpected}")
        if set(unexpected) & {"norm.weight", "norm.bias"} not in (set(), {"norm.weight", "norm.bias"}):
            raise ValueError("RETFound checkpoint contains an incomplete encoder norm")
        report.update(missing_keys=missing, unexpected_keys=unexpected,
                      default_norm="fc_norm LayerNorm ones/zeros, eps=1e-6" if norm <= set(missing) else None)
        transform = transforms.Compose([transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
                                        transforms.CenterCrop(224), transforms.ToTensor(),
                                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        prepare = lambda images: torch.stack([transform(im.convert("RGB")) for im in images])

        def forward(x):
            values = model.forward_features(x)
            if values.shape != (len(x), 1, 1024):
                raise ValueError("RETFound forward_features must return [batch, 1, 1024]")
            return values[:, 0, :]
    elif arm == "MIRAGE":
        from safetensors.torch import load_file
        source = import_source(assets / "mirage_hf.py")
        config = prep.read_json(assets / "config.json")
        if config != dict(input_size=512, modalities="bscan-slo", patch_size=32, size="base"):
            raise ValueError("unexpected MIRAGE-Base architecture config")
        model = source.MIRAGEWrapper(**config)
        state = load_file(str(assets / "model.safetensors"))
        if set(state) == set(model.state_dict()):
            torch.nn.Module.load_state_dict(model, state, strict=True)
            report["checkpoint_layout"] = "wrapper"
        elif set(state) == set(model.model.state_dict()):
            model.model.load_state_dict(state, strict=True)
            report["checkpoint_layout"] = "encoder"
        else:
            raise ValueError("MIRAGE state keys differ from both exact upstream layouts")
        norm = torch.nn.LayerNorm(768, eps=1e-6).float().cuda().eval().requires_grad_(False)
        report["default_norm"] = "classification-wrapper LayerNorm ones/zeros, eps=1e-6, not in checkpoint"
        prepare = lambda images: torch.stack([torch.from_numpy(mirage_input(im))[None] for im in images])

        def forward(x):
            values = model({"bscan": x})
            if values.shape != (len(x), 257, 768) or model.args.num_global_tokens != 1:
                raise ValueError("MIRAGE OCT-only output must have 256 patches and one final global token")
            return norm(values)[:, :-1, :].mean(dim=1)
    else:
        # Construct only the image tower. OpenCLIP imports include an unused local tokenizer.
        from open_clip.timm_model import TimmModel
        from open_clip.transform import PreprocessCfg, image_transform_v2
        config = prep.read_json(assets / "open_clip_config.json")
        vision = config["model_cfg"]["vision_cfg"]
        if vision != dict(timm_model_name="vit_base_patch16_224", timm_model_pretrained=False,
                          timm_pool="", timm_proj="linear", image_size=224) or config["model_cfg"]["embed_dim"] != 512:
            raise ValueError("BioMedCLIP visual architecture differs from the pinned image tower")
        if config["preprocess_cfg"] != dict(mean=[.48145466, .4578275, .40821073],
                                            std=[.26862954, .26130258, .27577711]):
            raise ValueError("BioMedCLIP preprocessing differs from the pinned native recipe/geometry")
        model = TimmModel(model_name=vision["timm_model_name"], embed_dim=config["model_cfg"]["embed_dim"],
                          image_size=vision["image_size"], pool=vision["timm_pool"], proj=vision["timm_proj"], pretrained=False)
        state = torch.load(assets / "open_clip_pytorch_model.bin", map_location="cpu", weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]  # open_clip.factory.load_state_dict's explicit wrapper.
        visual = {k.removeprefix("visual."): v for k, v in state.items() if k.startswith("visual.")}
        model.load_state_dict(visual, strict=True)
        transform = image_transform_v2(PreprocessCfg(size=224, **config["preprocess_cfg"]), is_train=False)
        prepare = lambda images: torch.stack([transform(im) for im in images])
        forward = model
        report["text_tower"] = "not constructed; all visual weights strict-loaded"
    model.float().cuda().eval().requires_grad_(False)

    def encode(images):
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            result = forward(prepare(images).to(device="cuda", dtype=torch.float32))
            if result.dtype != torch.float32:
                raise ValueError("model output is not FP32")
            return result.cpu().numpy()

    return Encoder(arm, encode, torch.cuda.synchronize, report,
                   lambda: dict(peak_vram_bytes=torch.cuda.max_memory_allocated()))


REPEAT_TOLERANCE = 1e-4  # R3 (approved 11 Nov) and A2 A-6: largest |a - b| over largest |a|
REPEAT_COSINE = 0.9999  # A2 A-6: per real timing image, first versus last timed pass


def synthetic_images() -> list[Image.Image]:
    """R3's five fixed synthetic RGB images, in benchmark crop and Kermany scan sizes."""
    rng = fm.stream(fm.SEED, 9201)
    sizes = ((512, 496), (768, 496), (1536, 496), (414, 413), (327, 326))
    return [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB") for w, h in sizes]


def repeat_check(first: Sequence[np.ndarray], second: Sequence[np.ndarray]) -> dict:
    """Agreement of two passes over the same images: whole-set and per-image ratios, cosines."""
    a = np.concatenate(first).astype(np.float64)
    b = np.concatenate(second).astype(np.float64)
    if a.ndim != 2 or a.shape != b.shape or not len(a):
        raise ValueError("repeat check needs two aligned nonempty output sets")
    difference = np.abs(a - b)
    cosine = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return dict(images=len(a), max_difference_ratio=float(difference.max() / np.abs(a).max()),
                worst_image_difference_ratio=float((difference.max(axis=1) / np.abs(a).max(axis=1)).max()),
                min_cosine=float(cosine.min()))


def check_numerical(numerical: dict) -> None:
    """R3 on the synthetic set; A2 A-6 on every real timing image. Failure stops the pilot."""
    try:
        real, synthetic = numerical["real"], numerical["synthetic"]
        valid = numerical["tolerance"] == REPEAT_TOLERANCE and numerical["min_cosine"] == REPEAT_COSINE and \
            real["images"] == 100 and synthetic["images"] == 5
        passed = real["worst_image_difference_ratio"] <= REPEAT_TOLERANCE and real["min_cosine"] >= REPEAT_COSINE and \
            synthetic["max_difference_ratio"] <= REPEAT_TOLERANCE
    except (KeyError, TypeError):
        raise ValueError("pilot lacks the numerical repeat record") from None
    if not valid:
        raise ValueError("pilot numerical record does not implement the fixed R3/A-6 rule")
    if not passed:
        raise ValueError(f"numerical repeat check failed (R3/A2 A-6); report the arm unavailable: "
                         f"{json.dumps(numerical, sort_keys=True)}")


def warm_timing(encoder: Encoder, images: Sequence[Image.Image]) -> dict:
    """One warm pass + five timed passes, then the R3/A-6 repeat checks; validation is untimed.

    The first and last timed passes are compared on all 100 images (A2 A-6); five synthetic
    images are encoded twice after the timed passes (R3).
    """
    if len(images) != 100:
        raise ValueError("the timing inventory must contain exactly 100 realized IDs")
    started = time.monotonic_ns()
    per_pass, kept = [], {}
    validation_seconds = 0.
    for pass_index in range(6):
        outputs = []
        encoder.synchronize()
        start = time.perf_counter()
        for image in images:
            outputs.append(encoder.encode([image]))
        encoder.synchronize()
        elapsed = time.perf_counter() - start
        validation_start = time.perf_counter()
        for values in outputs:
            validate_features(values, 1, DIMENSIONS[encoder.arm])
        validation_seconds += time.perf_counter() - validation_start
        if pass_index:
            per_pass.append(elapsed / len(images))
        if pass_index in (1, 5):
            kept[pass_index] = outputs
    check_start = time.perf_counter()
    synthetic = synthetic_images()
    repeats = [[encoder.encode([image]) for image in synthetic] for _ in range(2)]
    for values in (*repeats[0], *repeats[1]):
        validate_features(values, 1, DIMENSIONS[encoder.arm])
    numerical = dict(rule="R3 synthetic repeat; A2 A-6 first-versus-last timed pass", tolerance=REPEAT_TOLERANCE,
                     min_cosine=REPEAT_COSINE, real=repeat_check(kept[1], kept[5]), synthetic=repeat_check(*repeats))
    validation_seconds += time.perf_counter() - check_start
    check_numerical(numerical)
    return dict(batch_size=1, images=100, warm_passes=1, timed_passes=5,
                seconds_per_image=per_pass, median_seconds_per_image=float(np.median(per_pass)),
                started_monotonic_ns=started, finished_monotonic_ns=time.monotonic_ns(),
                process_id=os.getpid(), validation_seconds=validation_seconds, numerical=numerical,
                includes="preprocessing, inference and host output transfer; excludes validation, "
                         "repeat checks, decoding, disk and download")


def validate_timing(timing: dict) -> None:
    """Reject incomplete, nonfinite or malformed pilot measurements on resume/gating."""
    if any(timing[key] != value for key, value in
           dict(batch_size=1, images=100, warm_passes=1, timed_passes=5).items()):
        raise ValueError("pilot does not implement the fixed timing recipe")
    seconds = np.asarray(timing["seconds_per_image"], dtype=np.float64)
    if seconds.shape != (5,) or not np.isfinite(seconds).all() or (seconds <= 0).any() or \
            timing["median_seconds_per_image"] != float(np.median(seconds)):
        raise ValueError("invalid pilot timing")
    if not 0 < timing["started_monotonic_ns"] < timing["finished_monotonic_ns"] or timing["process_id"] <= 0:
        raise ValueError("pilot needs a positive monotonic measurement window and process ID")
    check_numerical(timing.get("numerical"))


def peak_ram_bytes() -> int | None:
    if platform.system() != "Linux":
        return None
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


GEOMETRY_HEADER = ["path", "width", "height", "resize_width", "resize_height", "crop_left", "crop_top",
                   "crop_width", "crop_height", "retained_fraction", "border_share", "near_white_share"]


def operation_files(step: str) -> list[str]:
    base = ["identity.json", "load_report.json", "geometry.csv"]
    return base + (["timing.json"] if step == "pilot" else ["features.npz", "costs.json"])


def verify_operation(out: Path, step: str, inventory: prep.Rows, arm: str) -> None:
    """A final measurement is also the receipt for its inputs and completed artifacts."""
    files = operation_files(step)
    payload = prep.read_json(out / files[-1])
    if set(payload["files"]) != set(files[:-1]):
        raise ValueError("operation receipt has missing or unexpected artifacts")
    prep.verify_files(out, payload["files"])
    if step == "pilot":
        validate_timing(payload)
        if payload["image_ids"] != [row["path"] for row in inventory]:
            raise ValueError("pilot realized IDs differ from the frozen timing inventory")
        if not 0 < payload["operation_started_monotonic_ns"] <= payload["started_monotonic_ns"] < \
                payload["finished_monotonic_ns"] <= payload["operation_finished_monotonic_ns"]:
            raise ValueError("pilot operation window does not contain its timing passes")
        costs = payload["costs"]
    else:
        load_features(out / "features.npz", inventory, DIMENSIONS[arm])
        costs = payload
    if costs["image_count"] != len(inventory):
        raise ValueError("cost image count differs from the frozen inventory")
    for key in ("model_loading_seconds", "staging_seconds", "extraction_seconds", "validation_seconds",
                "serialization_seconds", "input_verification_seconds", "wall_seconds"):
        if not np.isfinite(costs[key]) or costs[key] < 0:
            raise ValueError(f"invalid operation cost: {key}")


def run(step: str, root: Path, folder: Path, spec: dict, assets: Path, out: Path, session: str,
        external: Path | None = None) -> None:
    require_colab()
    if step not in ("pilot", "extract"):
        raise ValueError("unknown extraction step")
    if external is not None and step != "extract":
        raise ValueError("external queries are extracted only under the input and query locks")
    if step == "pilot" and not session.strip():
        raise ValueError("a shared Colab timing-session ID is required")
    operation_started = time.monotonic_ns()
    wall_start = time.perf_counter()
    out = runtime_out(out)
    check_spec(spec)
    summary = prepared(folder)
    lock = prep.load_lock(folder) if step == "extract" else None
    arm = spec["arm"]
    input_rows = prep.read_csv(folder / "extraction_inventory.csv")
    if external is not None:
        from fm_external import verify_external  # fm_external imports this module
        verify_external(external, folder)
        input_rows = prep.read_csv(external / "external_extraction_inventory.csv")
    if lock is not None and lock["arms"].get(arm, {}).get("status") != "available":
        raise ValueError("arm was unavailable at input lock and cannot enter v1 later")
    current = dict(spec=spec, code=prep.code_identity(), environment=environment())
    if lock is not None and current != lock["arms"][arm]["identity"]:
        raise ValueError("extraction model/code/environment differs from the input lock")
    inventory = prep.read_csv(folder / "timing_inventory.csv") if step == "pilot" else input_rows
    validate_inventory([r["path"] for r in input_rows], [r["sha256"] for r in input_rows])
    if step == "pilot" and len(inventory) != 100:
        raise ValueError("the timing inventory must contain exactly 100 realized IDs")
    # Timing IDs may repeat, but staging integrity still checks every source used.
    content = {r["path"]: r["sha256"] for r in input_rows}
    verification_start = time.perf_counter()
    for path in {r["path"] for r in inventory}:
        if path not in content:
            raise ValueError(f"timing ID is absent from the extraction inventory: {path}")
        if prep.sha256(root / prep.safe_id(path)) != content[path]:
            raise ValueError(f"input image content differs: {path}")
    input_verification_seconds = time.perf_counter() - verification_start
    identity = dict(**current, step=step, input_files=summary["files"],
                    preparation_sha256=prep.sha256(folder / "freeze_summary.json"),
                    lock_sha256=prep.sha256(folder / "input_lock.json") if lock else None,
                    external_lock_sha256=prep.sha256(external / "external_lock.json") if external else None,
                    timing_session=session if step == "pilot" else None, hardware=hardware())
    with prep.writer_lock(out):
        prep.write_json(out / "identity.json", identity)
        files = operation_files(step)
        if (out / "complete.json").exists():
            marker = prep.verify_complete(out)
            if set(marker["files"]) != set(files):
                raise ValueError("completion marker has missing or unexpected artifacts")
            check_spec(spec, assets)
            verify_operation(out, step, inventory, arm)
            return
        if (out / files[-1]).exists():
            # Interrupted after the final measurement but before complete.json.
            check_spec(spec, assets)
            verify_operation(out, step, inventory, arm)
            prep.complete(out, files)
            return
        start = time.perf_counter()
        encoder = load_encoder(spec, assets)
        encoder.synchronize()
        loading_seconds = time.perf_counter() - start
        serialization_seconds = 0.
        start = time.perf_counter()
        prep.write_json(out / "load_report.json", encoder.load_report)
        serialization_seconds += time.perf_counter() - start
        images, geometries = [], []
        staging_seconds, extraction_seconds, validation_seconds = 0., 0., 0.
        if step == "pilot":
            start = time.perf_counter()
            for row in inventory:
                path = root / row["path"]
                if encoder.decode_path:
                    decoded = encoder.decode_path(path)
                else:
                    with Image.open(path) as image:
                        decoded = image.copy()
                geometries.append(geometry(arm, row["path"], decoded))
                images.append(decoded)
            staging_seconds = time.perf_counter() - start
            start = time.perf_counter()
            payload = warm_timing(encoder, images)
            extraction_seconds = time.perf_counter() - start - payload["validation_seconds"]
            validation_seconds = payload["validation_seconds"]
            payload.update(image_ids=[r["path"] for r in inventory],
                           operation_started_monotonic_ns=operation_started)
        else:
            features = np.empty((len(inventory), DIMENSIONS[arm]), dtype=np.float32)
            for i, row in enumerate(inventory):
                path = root / row["path"]
                start = time.perf_counter()
                with Image.open(path) as image:
                    geometries.append(geometry(arm, row["path"], image))
                    if encoder.stage_path is None:
                        decoded = image.copy()
                if encoder.stage_path:
                    infer = encoder.stage_path(path)
                else:
                    infer = lambda: encoder.encode([decoded])
                encoder.synchronize()
                staging_seconds += time.perf_counter() - start
                start = time.perf_counter()
                values = infer()
                encoder.synchronize()
                extraction_seconds += time.perf_counter() - start
                start = time.perf_counter()
                validate_features(values, 1, DIMENSIONS[arm])
                features[i] = values[0]
                validation_seconds += time.perf_counter() - start
                if (i + 1) % 100 == 0:
                    print(f"{arm}: {i + 1}/{len(inventory)}", flush=True)
            start = time.perf_counter()
            saving_seconds = save_features(out / "features.npz", [r["path"] for r in input_rows],
                                           [r["sha256"] for r in input_rows], features)
            validation_seconds += time.perf_counter() - start - saving_seconds
            serialization_seconds += saving_seconds
            start = time.perf_counter()
            load_features(out / "features.npz", input_rows, DIMENSIONS[arm])
            validation_seconds += time.perf_counter() - start
        start = time.perf_counter()
        prep.write_csv(out / "geometry.csv", GEOMETRY_HEADER, geometries)
        serialization_seconds += time.perf_counter() - start
        start = time.perf_counter()
        artifact_hashes = {name: prep.sha256(out / name) for name in files[:-1]}
        validation_seconds += time.perf_counter() - start
        costs = dict(model_loading_seconds=loading_seconds, input_verification_seconds=input_verification_seconds,
                     staging_seconds=staging_seconds, extraction_seconds=extraction_seconds,
                     validation_seconds=validation_seconds, serialization_seconds=serialization_seconds,
                     wall_seconds=time.perf_counter() - wall_start, download_seconds=None,
                     download_note="assets staged separately; unmeasured here",
                     model_loading_note="includes asset SHA256 verification",
                     staging_note="decode/copy; Keras full extraction also includes stage3.load_image preprocessing",
                     extraction_note="preprocessing/inference/output transfer; Keras full extraction is inference/output transfer",
                     serialization_note="payload arrays/load report/geometry; final measurement and marker writes excluded",
                     peak_ram_bytes=peak_ram_bytes(), image_count=len(inventory), **encoder.peak_memory())
        if step == "pilot":
            payload["costs"] = costs
            payload["operation_finished_monotonic_ns"] = time.monotonic_ns()
            validate_timing(payload)
        else:
            payload = costs
        payload["files"] = artifact_hashes
        prep.write_json(out / files[-1], payload)
        prep.complete(out, files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="step", required=True)
    describe = sub.add_parser("describe")
    describe.add_argument("--arm", choices=REPOSITORIES, required=True)
    spec = sub.add_parser("spec")
    spec.add_argument("--arm", choices=REPOSITORIES, required=True)
    spec.add_argument("--assets", type=Path, required=True)
    spec.add_argument("--revision", required=True)
    spec.add_argument("--source-revision", default="")
    spec.add_argument("--terms-evidence", required=True)
    spec.add_argument("--out", type=Path, required=True)
    for step in ("pilot", "extract"):
        command = sub.add_parser(step)
        for name in ("root", "prepared", "spec", "assets", "out"):
            command.add_argument("--" + name, type=Path, required=True)
        command.add_argument("--timing-session", default="", help="pilot only: shared ID for one sequential Colab session")
        if step == "extract":
            command.add_argument("--external", type=Path, help="external query-lock folder (fm_external.py freeze)")
    args = parser.parse_args()
    if args.step == "describe":
        print(json.dumps(recipe(args.arm), indent=2))
    elif args.step == "spec":
        prep.writable_path(prep.SCRIPT_ROOT,args.out)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        prep.write_json(args.out, make_spec(args.arm, args.assets, args.revision, args.source_revision, args.terms_evidence))
    else:
        run(args.step, args.root, args.prepared, prep.read_json(args.spec), args.assets, args.out, args.timing_session,
            getattr(args, "external", None))


if __name__ == "__main__":
    main()
