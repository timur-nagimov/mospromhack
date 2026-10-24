"""Экспорт ONNX: метаданные пишутся, даже если какого-то пакета из списка версий нет в окружении."""

from __future__ import annotations

import importlib.metadata

from ml import export_onnx


def test_missing_package_gives_none_instead_of_crash(monkeypatch):
    def version(package):
        if package == "opencv-python":
            raise importlib.metadata.PackageNotFoundError(package)
        return "1.0"

    monkeypatch.setattr(export_onnx.importlib.metadata, "version", version)
    assert export_onnx.package_versions(("onnx", "opencv-python")) == {"onnx": "1.0", "opencv-python": None}
