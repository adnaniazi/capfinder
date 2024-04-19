"""
The module contains constants used in the capfinder package.

Author: Adnan M. Niazi
Date: 2024-02-28
"""

CIGAR_CODES = ["M", "I", "D", "N", "S", "H", "P", "=", "X"]
CODE_TO_OP = {
    "M": 0,
    "I": 1,
    "D": 2,
    "N": 3,
    "S": 4,
    "H": 5,
    "P": 6,
    "=": 7,
    "X": 8,
}
OP_TO_CODE = {str(v): k for k, v in CODE_TO_OP.items()}
