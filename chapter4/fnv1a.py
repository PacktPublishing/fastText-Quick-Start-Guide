"""
➜  python fnv1a.py
13599670572911738159928974088951085519558194506275379142610729630040
"""

def fnv1a_32(string, seed=0):
    """
    Returns: The FNV-1a (alternate) hash of a given string
    """
    #Constants
    FNV_prime = 16777619
    offset_basis = 2166136261

    #FNV-1a Hash Function
    hash = offset_basis + seed
    for char in string:
        hash = hash ^ ord(char)
        hash = hash * FNV_prime
    return hash

if __name__ == "__main__":
    print(fnv1a_32('fasttext'))
