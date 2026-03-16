from typing import Tuple, Optional, Dict, List
import numpy as np  


def find_last_pattern(view: memoryview, pattern=b"TPX3") -> int:
    """
    Find the last occurrence of b"TPX3" in a memoryview.

    Args:
        view: memoryview of the data

    Returns:
        int: Index of the last occurrence of b"TPX3", or -1 if not found.
    """
    pattern_len = len(pattern)
    view_len = len(view)

    # Iterate from the end of the view backward
    for i in range(view_len - pattern_len, -1, -1):
        if view[i:i + pattern_len] == pattern:
            return i

    return -1