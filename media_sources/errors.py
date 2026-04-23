from __future__ import annotations


class ConfigError(Exception):
    pass


class SourceError(Exception):
    pass


class SourceOpenError(SourceError):
    pass


class SourceReadError(SourceError):
    pass
