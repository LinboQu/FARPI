from __future__ import annotations


def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v != 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "y", "t"}:
            return True
        if s in {"false", "0", "no", "n", "f"}:
            return False
    raise ValueError(f"Cannot parse bool from value: {v!r}")


def get_bool(cfg: dict, key: str, default: bool):
    if cfg is None:
        return bool(default)
    if key not in cfg:
        return bool(default)
    v = cfg.get(key)
    if v is None:
        return bool(default)
    if isinstance(v, str):
        try:
            return str2bool(v)
        except ValueError as e:
            raise ValueError(f"Invalid boolean for key '{key}': {v!r}") from e
    if isinstance(v, (bool, int)):
        return str2bool(v)
    return bool(v)


def get_float(cfg: dict, key: str, default: float):
    if cfg is None or key not in cfg or cfg.get(key) is None:
        return float(default)
    v = cfg.get(key)
    if isinstance(v, bool):
        return float(int(v))
    return float(v)


def get_int(cfg: dict, key: str, default: int):
    if cfg is None or key not in cfg or cfg.get(key) is None:
        return int(default)
    v = cfg.get(key)
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, str):
        return int(float(v.strip()))
    return int(v)


def get_str(cfg: dict, key: str, default: str):
    if cfg is None or key not in cfg or cfg.get(key) is None:
        return str(default)
    return str(cfg.get(key))
