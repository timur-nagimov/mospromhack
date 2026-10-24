"""Download pinned public ONNX/data; inspect numeric graphs without remote code.

No PyTorch, pickle, Hub scripts, model training or default-registry mutation.
Run as ``python -m tools.prepare_ppe_candidate --help`` from the repository root.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import stat
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REPO = "constructelligence/construction-site-safety-hazards"
REVISION = "79673cff5a8bb249fc8ec9cca3daa19e2ef95191"
MODEL_NAME = "buildvision-hazards-v0.3.onnx"
SOURCE_CLASSES = ["person", "hardhat", "no-hardhat", "safety vest", "no-safety vest",
                  "no-mask", "gloves", "safety shoes", "excavator", "wheel loader",
                  "dump truck", "goggles", "no-goggles", "no-gloves"]
CLASS_MAP = {"hardhat": "helmet", "no-hardhat": "no_helmet",
             "safety vest": "vest", "no-safety vest": "no_vest"}
CLASSES = [CLASS_MAP.get(name, name) for name in SOURCE_CLASSES]
FILES = {
    MODEL_NAME: (44_766_819, "55f4dd2f53158eac4ff093d938b9fbda80c99454a03672145bbc36f539988249"),
    "config.json": (560, "1c5ca99d2b5f6257668d13ab53761a82d4a4d8872488b4fadede63a57a058224"),
    "README.md": (13_065, "7091675890c21fa3be69bafa9526b606fb4edd570e1167d5b0d50a68ee4ea10a"),
}


@dataclass(frozen=True)
class CandidateProfile:
    name: str
    repository: str
    revision: str
    model_name: str
    directory: str
    files: dict[str, tuple[int, str]]
    source_classes: list[str]
    class_map: dict[str, str]
    hub_license: str
    registry: str
    source_config: str | None = None

    @property
    def classes(self) -> list[str]:
        return [self.class_map.get(name, name) for name in self.source_classes]


PROFILES = {
    "buildvision": CandidateProfile(
        "buildvision", REPO, REVISION, MODEL_NAME, "buildvision-hazards-v0.3", FILES,
        SOURCE_CLASSES, CLASS_MAP, "agpl-3.0", "models.ppe-candidate.yaml", "config.json"),
    "vyra": CandidateProfile(
        "vyra", "Hexmon/vyra-yolo-ppe-detection", "08895b33d95d2587423ebe4f7c1b9c41beebd642",
        "best.onnx", "vyra-yolo-ppe-detection", {
            "best.onnx": (103_655_282, "b99fed37bae1d111ddb622a0ef9077d42918a4daebd4a0bffbd8faaba273f246"),
            "README.md": (1578, "ec237e6af27e6a279f6e81afc7d387e6364e6c06739fe52143e2609662263458"),
            "args.yaml": (1645, "3117297c8c15c0e0b49dc9e9fb0c29113956ae8cedae917d8ee9b2094d9c5f82"),
        },
        ["Fall-Detected", "Gloves", "Goggles", "Hardhat", "Ladder", "Mask", "NO-Gloves",
         "NO-Goggles", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Person", "Safety Cone", "Safety Vest"],
        {"Hardhat": "helmet", "NO-Hardhat": "no_helmet", "NO-Safety Vest": "no_vest",
         "Person": "person", "Safety Vest": "vest"},
        "cc-by-4.0", "models.ppe-vyra.yaml"),
}


def get_profile(name: str = "buildvision") -> CandidateProfile:
    if name not in PROFILES:
        raise ValueError(f"Unknown candidate profile: {name}")
    return PROFILES[name]


DATA_REVISION = "42044baf5eda3d52aa4bd7d8a098375d2c7fdad7"
DATA_YAML_URL = (f"https://raw.githubusercontent.com/ultralytics/ultralytics/{DATA_REVISION}/"
                 "ultralytics/cfg/datasets/construction-ppe.yaml")
DATA_YAML_SHA = "8926b6ec2c28c2fd6fc9c54afe8de12bc517c0713c11d35323a36f0d50729dc9"
DATA_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
DATA_BYTES = 178_415_813
DATA_SHA = "bef8dcb599aa4e9d9f5e602cb6fa7143d3c84d7f6a0ff40463d7f2a4c2632ccc"
DATA_NAMES = ["helmet", "gloves", "vest", "boots", "goggles", "none", "Person",
              "no_helmet", "no_goggle", "no_gloves", "no_boots"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def asset(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), f"Not a regular file: {path}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def download(url: str, destination: Path, *, max_bytes: int, expected_sha: str | None = None,
             expected_bytes: int | None = None) -> dict:
    """Bounded HTTPS download, atomic publication, verified immutable cache."""
    require(url.startswith("https://"), "HTTPS source required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.is_symlink(), f"Symlink destination: {destination}")
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        require(not temporary.is_symlink(), f"Symlink temporary file: {temporary}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "guard-ppe-candidate/1"})
            with urllib.request.urlopen(req, timeout=60) as response, temporary.open("wb") as out:
                require(response.geturl().startswith("https://"), "Non-HTTPS redirect")
                length = response.headers.get("Content-Length")
                require(length is None or int(length) <= max_bytes, "Download size exceeds limit")
                count = 0
                while block := response.read(1 << 20):
                    count += len(block)
                    require(count <= max_bytes, "Download size exceeds limit")
                    out.write(block)
            info = asset(temporary)
            require(expected_bytes is None or info["bytes"] == expected_bytes, "Unexpected download size")
            require(expected_sha is None or info["sha256"] == expected_sha, "Download SHA-256 mismatch")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    info = asset(destination)
    require(info["bytes"] <= max_bytes, "Cached file exceeds limit")
    require(expected_bytes is None or info["bytes"] == expected_bytes, "Cached size mismatch")
    require(expected_sha is None or info["sha256"] == expected_sha, "Cached SHA-256 mismatch")
    return {"url": url, **info}


def inspect_onnx(path: Path, source_config: Path | None = None, candidate_profile: str = "buildvision") -> dict:
    import onnx

    profile = get_profile(candidate_profile)
    require(sha256(path) == profile.files[profile.model_name][1], "Unexpected ONNX SHA-256")
    if profile.source_config is not None:
        require(source_config is not None, "Source config required for this profile")
        config = json.loads(source_config.read_text(encoding="utf-8"))
        require(config["classes"] == profile.source_classes and config["imgsz"] == 640, "Source config drift")
    model = onnx.load(str(path), load_external_data=False)
    require(not any(t.external_data or t.data_location == onnx.TensorProto.EXTERNAL
                    for t in model.graph.initializer), "External tensor data is not allowed")
    require(not model.functions, "Local ONNX functions are outside the reviewed candidate")
    require(all(node.domain in ("", "ai.onnx") for node in model.graph.node), "Custom operators are not allowed")
    require(not any(node.op_type == "NonMaxSuppression" for node in model.graph.node), "Embedded NMS is unsupported")
    onnx.checker.check_model(model, full_check=True)

    def tensor(value):
        return {"name": value.name, "dtype": int(value.type.tensor_type.elem_type),
                "shape": [dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param
                          for dimension in value.type.tensor_type.shape.dim]}

    inputs, outputs = [tensor(v) for v in model.graph.input], [tensor(v) for v in model.graph.output]
    require(len(inputs) == 1 and inputs[0]["shape"] == [1, 3, 640, 640]
            and inputs[0]["dtype"] == onnx.TensorProto.FLOAT, "Unexpected model input")
    require(len(outputs) == 1 and outputs[0]["shape"] == [1, 4 + len(profile.classes), 8400]
            and outputs[0]["dtype"] == onnx.TensorProto.FLOAT, "Output/class-count mismatch")
    metadata = {p.key: p.value for p in model.metadata_props}
    names = ast.literal_eval(metadata["names"])
    require(names == dict(enumerate(profile.source_classes)), "ONNX/profile class-order mismatch")
    require("AGPL-3.0" in metadata.get("license", ""), "Unexpected ONNX license metadata")
    return {"checker": "passed", "inputs": inputs, "outputs": outputs,
            "source_classes": profile.source_classes, "classes": profile.classes, "class_map": profile.class_map,
            "opsets": {o.domain: o.version for o in model.opset_import}, "metadata": metadata}


def prepare_model(models_dir: Path, numerical_smoke: bool = False, candidate_profile: str = "buildvision") -> dict:
    profile = get_profile(candidate_profile)
    directory = models_dir / "candidates" / profile.directory
    api_url = f"https://huggingface.co/api/models/{profile.repository}/revision/{profile.revision}?blobs=true"
    download(api_url, directory / "hub-metadata.json", max_bytes=1 << 20)
    hub = json.loads((directory / "hub-metadata.json").read_text())
    require(hub["id"] == profile.repository and hub["sha"] == profile.revision, "Unexpected Hub repository revision")
    require(hub["cardData"]["license"] == profile.hub_license, "Unexpected repository license")
    sibling = next(item for item in hub["siblings"] if item["rfilename"] == profile.model_name)
    model_bytes, model_sha = profile.files[profile.model_name]
    require(sibling["size"] == model_bytes and sibling["lfs"]["sha256"] == model_sha, "Unexpected Hub ONNX metadata")
    records = []
    for filename, (size, digest) in profile.files.items():
        local_name = {"README.md": "README.source.md", "args.yaml": "args.source.yaml"}.get(filename, filename)
        records.append(download(f"https://huggingface.co/{profile.repository}/resolve/{profile.revision}/{filename}",
                                directory / local_name, max_bytes=size, expected_bytes=size, expected_sha=digest))
    inspection = inspect_onnx(directory / profile.model_name,
                              directory / profile.source_config if profile.source_config else None, profile.name)
    license_provenance = {"hub": profile.hub_license, "onnx_metadata": inspection["metadata"]["license"],
                          "status": "consistent" if profile.hub_license == "agpl-3.0" else "conflicting_metadata",
                          "review_required": profile.hub_license != "agpl-3.0"}
    license_label = "AGPL-3.0" if license_provenance["status"] == "consistent" else None
    sidecar = {"model_id": "ppe", "candidate_profile": profile.name,
               "onnx": profile.model_name, "onnx_path": profile.model_name, "input_size": [640, 640],
               "input_dtype": "float32", "output_format": "yolov8", **inspection,
               "supported_attributes": ["helmet", "vest"], "unsupported_attributes": ["respirator"],
               "source": f"https://huggingface.co/{profile.repository}", "revision": profile.revision,
               "license": license_label, "license_provenance": license_provenance,
               "status": "experimental", "sha256": model_sha,
               "onnx_sha256": model_sha, "onnx_bytes": model_bytes}
    write_json(directory / "ppe.json", sidecar)
    report = {"schema_version": 1, "status": "experimental", "candidate_profile": profile.name,
              "repository": profile.repository, "revision": profile.revision,
              "license": license_label, "license_provenance": license_provenance, "files": records,
              "hub_metadata": {"url": api_url, **asset(directory / "hub-metadata.json")}, "inspection": inspection,
              "numerical_smoke": {"status": "not_run"}, "training_overlap_with_evaluation": "unverified"}
    if numerical_smoke:
        import numpy as np
        from guard.contracts import ModelSpec
        from guard.inference.onnx_cpu import OnnxCpuBackend

        backend = OnnxCpuBackend(models_dir=models_dir, intra_op_threads=2)
        spec = ModelSpec(model_id="ppe", onnx_path=str(directory / profile.model_name), classes=profile.classes)
        try:
            backend.load(spec)
            outputs = backend.infer("ppe", np.zeros((1, 3, 640, 640), np.float32))
            require(len(outputs) == 1 and outputs[0].shape == (1, 4 + len(profile.classes), 8400), "Numerical shape mismatch")
            require(np.isfinite(outputs[0]).all(), "Non-finite numerical output")
            report["numerical_smoke"] = {"status": "passed", "provider": "CPUExecutionProvider",
                                          "threads": 2, "input": "zeros", "quality_evidence": False}
        finally:
            backend.close()
    write_json(directory / "source.json", report)
    return report


def safe_extract_test(archive: Path, destination: Path) -> None:
    """Validate every entry; extract only test images/labels, never train/val."""
    require(not destination.exists(), "Extraction destination already exists")
    selected = []
    with zipfile.ZipFile(archive) as source:
        entries = source.infolist()
        require(len(entries) <= 10_000, "Too many ZIP entries")
        require(sum(item.file_size for item in entries) <= 600_000_000, "ZIP uncompressed-size limit")
        paths = set()
        for item in entries:
            name = item.filename
            path = PurePosixPath(name)
            require(name and "\\" not in name and "\x00" not in name and not path.is_absolute()
                    and ".." not in path.parts and all(":" not in part for part in path.parts), "Unsafe ZIP path")
            require(path not in paths, "Duplicate ZIP path")
            paths.add(path)
            mode = item.external_attr >> 16
            require(not stat.S_ISLNK(mode), "ZIP symlinks are not allowed")
            require(not (item.flag_bits & 1), "Encrypted ZIP entries are not allowed")
            require(item.file_size <= 20_000_000, "ZIP member-size limit")
            require(item.file_size <= max(item.compress_size, 1) * 200, "ZIP compression-ratio limit")
            parts = path.parts[1:] if path.parts[0] == "construction-ppe" else path.parts
            if len(parts) == 3 and parts[:2] in (("images", "test"), ("labels", "test")):
                suffix = path.suffix.lower()
                require((parts[0] == "images" and suffix in IMAGE_SUFFIXES)
                        or (parts[0] == "labels" and suffix == ".txt"), "Unexpected test member type")
                require(not item.is_dir() and (stat.S_IFMT(mode) in (0, stat.S_IFREG)), "Nonregular ZIP test member")
                selected.append((item, Path(*parts)))
        require(sum(relative.parts[0] == "images" for _, relative in selected) == 141, "Expected 141 test images")
        require(sum(relative.parts[0] == "labels" for _, relative in selected) == 141, "Expected 141 test labels")
        try:
            destination.mkdir(parents=True)
            for item, relative in selected:
                output = destination / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                with source.open(item) as inp, output.open("xb") as out:
                    shutil.copyfileobj(inp, out, 1 << 20)
        except Exception:
            shutil.rmtree(destination)
            raise


def prepare_dataset(directory: Path) -> dict:
    import yaml

    yaml_info = download(DATA_YAML_URL, directory / "construction-ppe.source.yaml", max_bytes=20_000,
                         expected_sha=DATA_YAML_SHA)
    meta = yaml.safe_load((directory / "construction-ppe.source.yaml").read_text())
    require(meta["names"] == dict(enumerate(DATA_NAMES)), "Official dataset class-order drift")
    require(meta["download"] == DATA_URL and meta["test"] == "images/test", "Official dataset source drift")
    zip_info = download(DATA_URL, directory / "construction-ppe.zip", max_bytes=DATA_BYTES,
                        expected_bytes=DATA_BYTES, expected_sha=DATA_SHA)
    extracted = directory / "test-only"
    if not extracted.exists():
        safe_extract_test(directory / "construction-ppe.zip", extracted)
    (extracted / "data.yaml").write_text((directory / "construction-ppe.source.yaml").read_text(), encoding="utf-8")
    require(not (extracted / "images" / "val").exists() and not (extracted / "images" / "train").exists(),
            "Only test split may be extracted")
    images = sorted((extracted / "images" / "test").glob("*"))
    labels = sorted((extracted / "labels" / "test").glob("*.txt"))
    require(len(images) == len(labels) == 141, "Expected exactly 141 image/label pairs")
    require({p.stem for p in images} == {p.stem for p in labels}, "Test image/label pair mismatch")
    records = [{"relative_path": str(p.relative_to(extracted)), **asset(p)} for p in [*images, *labels]]
    report = {"schema_version": 1, "source": "https://docs.ultralytics.com/datasets/detect/construction-ppe/",
              "license": "AGPL-3.0", "yaml_revision": DATA_REVISION, "source_yaml": yaml_info,
              "archive": zip_info, "split": "test", "images": 141, "labels": 141,
              "root": str(extracted.resolve()), "members": records,
              "training_overlap_with_candidate": "unverified"}
    write_json(directory / "source.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--candidate-profile", choices=tuple(PROFILES), default="buildvision")
    parser.add_argument("--data-dir", type=Path, default=Path("data/model_research/construction-ppe-test"))
    parser.add_argument("--dataset", action="store_true", help="also prepare the official test split")
    parser.add_argument("--numerical-smoke", action="store_true", help="one CPU-only zero-tensor inference, two threads")
    args = parser.parse_args(argv)
    model = prepare_model(args.models_dir.resolve(), args.numerical_smoke, args.candidate_profile)
    print(f"ONNX checker: {model['inspection']['checker']}; numerical: {model['numerical_smoke']['status']}")
    if args.dataset:
        data = prepare_dataset(args.data_dir.resolve())
        print(f"Dataset: {data['images']} test images at {data['root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
