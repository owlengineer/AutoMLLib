
def merge_dicts(dst: dict, src: dict):
    """update the values from other dictionary if key is present in other dictionary."""
    for sub in dst:
        if sub in src:
            dst[sub] = src[sub]
