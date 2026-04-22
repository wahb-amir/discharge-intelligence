def truncate_text(value, max_chars: int = 160) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text[:max_chars]


def safe_get(dct, *keys, default=None):
    cur = dct
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def take_first_unique(items, key: str, limit: int):
    seen = set()
    out = []
    for item in items:
        value = item.get(key, "Unknown")
        if value in seen:
            continue
        seen.add(value)
        out.append(item)
        if len(out) >= limit:
            break
    return out