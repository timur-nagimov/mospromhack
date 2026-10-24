"""Lightweight camera API errors shared by local and gateway engines."""


class CameraError(ValueError):
    """Invalid camera configuration."""


class CameraExists(KeyError):
    """Camera identifier is already configured."""


class CameraReadOnly(CameraError):
    """Gateway camera configuration is owned by its edge."""
