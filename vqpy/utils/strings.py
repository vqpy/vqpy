def longest_prefix_in(b: str, a: str):
    """Return the length of the longest prefix of b that appears in a"""
    # return the longest prefix of b that appears in a (for best match)
    l, r = 0, len(b)
    while l < r:
        m = (l + r + 1) >> 1
        if b[:m] in a: l = m
        else: r = m - 1
    return l