def longest_prefix_in(b: str, a: str):
    """Return the length of the longest prefix of b that appears in a"""
    # return the longest prefix of b that appears in a (for best match)
    left, right = 0, len(b)
    while left < right:
        mid = (left + right + 1) >> 1
        if b[:mid] in a:
            left = mid
        else:
            right = mid - 1
    return left
