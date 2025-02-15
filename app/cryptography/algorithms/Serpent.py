import os


def S(box, input):
    """Apply S-box number 'box' to a 4-bit byte and return a 4-bit byte as the result."""
    return int.from_bytes(SBoxBitstring[box % 8][bytes([input])])

def SInverse(box, output):
    """Apply S-box number 'box' in reverse to a 4-bit byte and return a 4-bit byte as the result."""
    return int.from_bytes(SBoxBitstringInverse[box % 8][bytes([output])])

def SHat(box, input):
    """Apply a parallel array of 32 copies of S-box number 'box' to the 16-byte input and return a 16-byte result."""
    result = bytearray(16)
    for i in range(32):
        result[i // 2] |= S(box, (input[i // 2] >> (4 * (i % 2))) & 0xF) << (4 * (i % 2))
    return bytes(result)

def SHatInverse(box, output):
    """Apply, in reverse, a parallel array of 32 copies of S-box number 'box' to the 16-byte output and return a 16-byte input."""
    result = bytearray(16)
    for i in range(32):
        result[i // 2] |= SInverse(box, (output[i // 2] >> (4 * (i % 2))) & 0xF) << (4 * (i % 2))
    return bytes(result)

def SBitslice(box, words):
    """Take 'words', a list of 4 4-byte arrays, least significant word first. Return a similar list of 4 4-byte arrays."""
    result = [bytearray(4) for _ in range(4)]
    for i in range(32):  # ideally in parallel
        quad = S(box, ((words[0][i // 8] >> (i % 8)) & 1) |
                      (((words[1][i // 8] >> (i % 8)) & 1) << 1) |
                      (((words[2][i // 8] >> (i % 8)) & 1) << 2) |
                      (((words[3][i // 8] >> (i % 8)) & 1) << 3))
        for j in range(4):
            result[j][i // 8] |= ((quad >> j) & 1) << (i % 8)
    return [bytes(r) for r in result]

def SBitsliceInverse(box, words):
    """Take 'words', a list of 4 4-byte arrays, least significant word first. Return a similar list of 4 4-byte arrays."""
    result = [bytearray(4) for _ in range(4)]
    for i in range(32):  # ideally in parallel
        quad = SInverse(box, ((words[0][i // 8] >> (i % 8)) & 1) |
                             (((words[1][i // 8] >> (i % 8)) & 1) << 1) |
                             (((words[2][i // 8] >> (i % 8)) & 1) << 2) |
                             (((words[3][i // 8] >> (i % 8)) & 1) << 3))
        for j in range(4):
            result[j][i // 8] |= ((quad >> j) & 1) << (i % 8)
    return [bytes(r) for r in result]

def LT(input):
    """Apply the table-based version of the linear transformation to the 16-byte input and return a 16-byte result."""
    if len(input) != 16:
        raise ValueError("input to LT is not 16 bytes long")

    result = bytearray(16)
    for i in range(len(LTTable)):
        outputBit = 0
        for j in LTTable[i]:
            outputBit ^= (input[j // 8] >> (j % 8)) & 1
        result[i // 8] |= outputBit << (i % 8)
    return bytes(result)

def LTInverse(output):
    """Apply the table-based version of the inverse of the linear transformation to the 16-byte output and return a 16-byte input."""
    if len(output) != 16:
        raise ValueError("input to inverse LT is not 16 bytes long")

    result = bytearray(16)
    for i in range(len(LTTableInverse)):
        inputBit = 0
        for j in LTTableInverse[i]:
            inputBit ^= (output[j // 8] >> (j % 8)) & 1
        result[i // 8] |= inputBit << (i % 8)
    return bytes(result)

def LTInverse(output):
    """Apply the table-based version of the inverse of the linear
    transformation to the 16-byte output and return a 16-byte input."""

    if len(output) != 16:
        raise ValueError("input to inverse LT is not 16 bytes long")

    result = bytearray(16)
    for i in range(len(LTTableInverse)):
        inputBit = 0
        for j in LTTableInverse[i]:
            inputBit ^= (output[j // 8] >> (j % 8)) & 1
        result[i // 8] |= inputBit << (i % 8)
    return bytes(result)


def LTBitslice(X):
    """Apply the equations-based version of the linear transformation to
    'X', a list of 4 4-byte arrays, least significant byte array first,
    and return another list of 4 4-byte arrays as the result."""
    
    X[0] = rotate_left(X[0], 13)
    X[2] = rotate_left(X[2], 3)
    X[1] = xor_bytes(X[1], X[0], X[2])
    X[3] = xor_bytes(X[3], X[2], shift_left(X[0], 3))
    X[1] = rotate_left(X[1], 1)
    X[3] = rotate_left(X[3], 7)
    X[0] = xor_bytes(X[0], X[1], X[3])
    X[2] = xor_bytes(X[2], X[3], shift_left(X[1], 7))
    X[0] = rotate_left(X[0], 5)
    X[2] = rotate_left(X[2], 22)

    return X

def LTBitsliceInverse(X):
    """Apply, in reverse, the equations-based version of the linear
    transformation to 'X', a list of 4 4-byte arrays, least significant
    byte array first, and return another list of 4 4-byte arrays as the
    result."""

    X[2] = rotate_right(X[2], 22)
    X[0] = rotate_right(X[0], 5)
    X[2] = xor_bytes(X[2], X[3], shift_left(X[1], 7))
    X[0] = xor_bytes(X[0], X[1], X[3])
    X[3] = rotate_right(X[3], 7)
    X[1] = rotate_right(X[1], 1)
    X[3] = xor_bytes(X[3], X[2], shift_left(X[0], 3))
    X[1] = xor_bytes(X[1], X[0], X[2])
    X[2] = rotate_right(X[2], 3)
    X[0] = rotate_right(X[0], 13)

    return X


def IP(input):
    """Apply the Initial Permutation to the 16-byte input
    and return a 16-byte result."""

    return applyPermutation(IPTable, input)

def FP(input):
    """Apply the Final Permutation to the 16-byte input
    and return a 16-byte result."""

    return applyPermutation(FPTable, input)


def IPInverse(output):
    """Apply the Initial Permutation in reverse."""
    
    return FP(output)

def FPInverse(output):
    """Apply the Final Permutation in reverse."""
    
    return IP(output)

def applyPermutation(permutationTable, input):
    """Apply the permutation specified by the 128-element list
    'permutationTable' to the 128-byte input and return a
    128-byte result."""

    if len(input) * 8 != len(permutationTable):
        raise ValueError("input size (%d) doesn't match perm table size (%d)" % (len(input) * 8, len(permutationTable)))
    result = bytearray(len(input))
    for i in range(len(permutationTable)):
        byte_index = permutationTable[i] // 8
        bit_index = permutationTable[i] % 8
        bit = (input[byte_index] >> bit_index) & 1
        result[i // 8] |= bit << (i % 8)
    return bytes(result)


def R(i, BHati, KHat):
    """Apply round 'i' to the 128-byte input 'BHati', returning another
    128-byte result. Do this using the appropriately numbered subkey(s) from the 'KHat' list of 33 128-byte keys."""

    xored = xor_bytes(BHati, KHat[i])

    SHati = SHat(i, xored)

    if 0 <= i <= r-2:
        BHatiPlus1 = LT(SHati)
    elif i == r-1:
        BHatiPlus1 = xor_bytes(SHati, KHat[r])
    else:
        raise ValueError("round %d is out of 0..%d range" % (i, r-1))

    return BHatiPlus1


def RInverse(i, BHatiPlus1, KHat):
    """Apply round 'i' in reverse to the 128-byte input 'BHatiPlus1',
    returning another 128-byte result. Do this using the appropriately numbered subkey(s) from the 'KHat' list of 33 128-byte keys."""


    if 0 <= i <= r-2:
        SHati = LTInverse(BHatiPlus1)
    elif i == r-1:
        SHati = xor_bytes(BHatiPlus1, KHat[r])
    else:
        raise ValueError("round %d is out of 0..%d range" % (i, r-1))

    xored = SHatInverse(i, SHati)

    BHati = xor_bytes(xored, KHat[i])

    return BHati

def RBitslice(i, Bi, K):
    """Apply round 'i' (bitslice version) to the 128-bit byte array 'Bi' and
    return another 128-bit byte array (conceptually B i+1). Use the
    appropriately numbered subkey(s) from the 'K' list of 33 128-bit
    byte arrays."""

    # 1. Key mixing
    xored = xor_bytes(Bi, K[i])

    # 2. S Boxes
    Si = SBitslice(i, quadSplit(xored))
    # Input and output to SBitslice are both lists of 4 32-bit byte arrays

    # 3. Linear Transformation
    if i == r-1:
        # In the last round, replaced by an additional key mixing
        BiPlus1 = xor_bytes(quadJoin(Si), K[r])
    else:
        BiPlus1 = quadJoin(LTBitslice(Si))
    # BIPlus1 is a 128-bit byte array

    return BiPlus1


def RBitsliceInverse(i, BiPlus1, K):
    """Apply the inverse of round 'i' (bitslice version) to the 128-bit
    byte array 'BiPlus1' and return another 128-bit byte array (conceptually
    B i). Use the appropriately numbered subkey(s) from the 'K' list of 33
    128-bit byte arrays."""

    # 3. Linear Transformation
    if i == r-1:
        # In the last round, replaced by an additional key mixing
        Si = quadSplit(xor_bytes(BiPlus1, K[r]))
    else:
        Si = LTBitsliceInverse(quadSplit(BiPlus1))
    # SOutput (same as LTInput) is a list of 4 32-bit byte arrays

    # 2. S Boxes
    xored = SBitsliceInverse(i, Si)
    # SInput and SOutput are both lists of 4 32-bit byte arrays

    # 1. Key mixing
    Bi = xor_bytes(quadJoin(xored), K[i])

    return Bi

def encrypt(userKey, plainText):
    """Encrypt the plaintext using CBC mode."""
    userKey = padKey(userKey, 32)
    plainText = pad(plainText)

    blocks = [plainText[i:i+16] for i in range(0, len(plainText), 16)]
    cipherText = b''

    # Initialization vector
    iv = os.urandom(16)

    prev = iv
    for block in blocks:
        block = xor_bytes(block, prev)
        cipher = encryptPart(block, userKey)
        cipherText += cipher
        prev = cipher

    return iv + cipherText

def decrypt(userKey, cipherText):
    """Decrypt the ciphertext using CBC mode."""
    userKey = padKey(userKey, 32)

    blocks = [cipherText[i:i+16] for i in range(16, len(cipherText), 16)]
    iv = cipherText[:16]
    plainText = b''

    prev = iv
    for block in blocks:
        decrypted = decryptPart(block, userKey)
        plainText += xor_bytes(decrypted, prev)
        prev = block

    return unpad(plainText)

def encryptPart(plainText, userKey):
    """Encrypt the 128-byte 'plainText' with the 32-byte 'userKey', using the normal algorithm, and return a 128-byte ciphertext."""
    K, KHat = makeSubkeys(userKey)

    BHat = IP(plainText) # BHat_0 at this stage
    for i in range(r):
        BHat = R(i, BHat, KHat) # Produce BHat_i+1 from BHat_i
    # BHat is now _32 i.e. _r
    C = FP(BHat)

    return C


def encryptBitslice(plainText, userKey):
    """Encrypt the 128-byte 'plainText' with the 32-byte 'userKey', using the bitslice algorithm, and return a 128-byte ciphertext."""
    K, KHat = makeSubkeys(userKey)

    B = plainText # B_0 at this stage
    for i in range(r):
        B = RBitslice(i, B, K) # Produce B_i+1 from B_i
    # B is now _r

    return B

def decryptPart(cipherText, userKey):
    """Decrypt the 128-byte 'cipherText' with the 32-byte 'userKey', using the normal algorithm, and return a 128-byte plaintext."""
    userKey = padKey(userKey, 32)
    K, KHat = makeSubkeys(userKey)

    BHat = FPInverse(cipherText) # BHat_r at this stage
    for i in range(r-1, -1, -1): # from r-1 down to 0 included
        BHat = RInverse(i, BHat, KHat) # Produce BHat_i from BHat_i+1
    # BHat is now _0
    plainText = IPInverse(BHat)

    return plainText
    

def decryptBitslice(cipherText, userKey):
    """Decrypt the 128-byte 'cipherText' with the 32-byte 'userKey', using the bitslice algorithm, and return a 128-byte plaintext."""
    K, KHat = makeSubkeys(userKey)

    B = cipherText # B_r at this stage
    for i in range(r-1, -1, -1): # from r-1 down to 0 included
        B = RBitsliceInverse(i, B, K) # Produce B_i from B_i+1
    # B is now _0

    return B


def makeSubkeys(userKey):
    """Given the 32-byte 'userKey', return two lists (conceptually K and KHat) of 33 16-byte keys each."""

    # We write the key as 8 4-byte words w-8 ... w-1
    w = {}  
    for i in range(-8, 0):
        w[i] = userKey[(i+8)*4:(i+9)*4]
        
    # We expand these to a prekey w0 ... w131 with the affine recurrence
    for i in range(132):
        w[i] = rotate_left(
            xor_bytes(w[i-8], w[i-5], w[i-3], w[i-1],
                        int_to_bytes(phi, 4), int_to_bytes(i, 4)),
            11)

    # The round keys are now calculated from the prekeys using the S-boxes
    # in bitslice mode. Each k[i] is a 4-byte array.
    k = {}
    for i in range(r+1):
        whichS = (r + 3 - i) % r
        k[0+4*i] = bytearray(4)
        k[1+4*i] = bytearray(4)
        k[2+4*i] = bytearray(4)
        k[3+4*i] = bytearray(4)
        for j in range(32): # for every bit in the k and w words
            input = ((w[0+4*i][j // 8] >> (j % 8)) & 1) | \
                    (((w[1+4*i][j // 8] >> (j % 8)) & 1) << 1) | \
                    (((w[2+4*i][j // 8] >> (j % 8)) & 1) << 2) | \
                    (((w[3+4*i][j // 8] >> (j % 8)) & 1) << 3)
            output = S(whichS, input)
            for l in range(4):
                k[l+4*i][j // 8] |= ((output >> l) & 1) << (j % 8)

    # We then renumber the 32-bit values k_j as 128-bit subkeys K_i.
    K = []
    for i in range(33):
        # ENOTE: k4i is the least significant word, k4i+3 the most.
        K.append(k[4*i] + k[4*i+1] + k[4*i+2] + k[4*i+3])

    # We now apply IP to the round key in order to place the key bits in
    # the correct column
    KHat = []
    for i in range(33):
        KHat.append(IP(K[i]))

    return K, KHat
def makeLongKey(k):
    """Take a key k in bytes format. Return the long version of that key."""

    l = len(k) * 8
    if l % 32 != 0 or l < 64 or l > 256:
        raise ValueError("Invalid key length (%d bits)" % l)
    
    if l == 256:
        return k
    else:
        return k + b'\x80' + b'\x00' * ((256 - l - 8) // 8)
        


def padKey(k, length):
    """Take a key k in bytes format and return a key of the specified length. use bit padding scheme"""
    if len(k) > length:
        raise ValueError("Key is too long (%d bytes)" % len(k))

    if len(k) == length:
        return k

    return k + b'\x01' + b'\x00' * (length - len(k) - 1)


def pad(plaintext):
    """
    Pads the given plaintext with PKCS#7 padding to a multiple of 16 bytes.
    Note that if the plaintext size is a multiple of 16, a whole block will be added.
    """
    plaintext = bytes(plaintext, encoding='utf-8')
    padding_len = 16 - (len(plaintext) % 16)
    padding = bytes([padding_len] * padding_len)
    return plaintext + padding

def unpad(plaintext):
    """
    Removes a PKCS#7 padding, returning the unpadded text and ensuring the
    padding was correct.
    """
    padding_len = plaintext[-1]
    assert padding_len > 0
    message, padding = plaintext[:-padding_len], plaintext[-padding_len:]
    assert all(p == padding_len for p in padding)
    return message

# --------------------------------------------------------------
# Generic bit-level primitives

# Internally, we represent the numbers manipulated by the cipher in a
# format that we call 'bitstring'. This is a string of "0" and "1"
# characters containing the binary representation of the number in
# little-endian format (so that subscripting with an index of i gives bit
# number i, corresponding to a weight of 2^i). This representation is only
# defined for nonnegative numbers (you can see why: think of the great
# unnecessary mess that would result from sign extension, two's complement
# and so on).  Example: 10 decimal is "0101" in bitstring format.

def int_to_bytes(n, length):
  """Convert an integer to a byte array of a given length."""
  return n.to_bytes(length, byteorder='little')

def bytes_to_int(b):
  """Convert a byte array to an integer."""
  return int.from_bytes(b, byteorder='little')

def binary_xor(b1, b2):
  """Return the XOR of two byte arrays of equal length."""
  return bytes(a ^ b for a, b in zip(b1, b2))

def xor_bytes(*args):
  """Return the XOR of a list of byte arrays."""
  result = args[0]
  for b in args[1:]:
    result = binary_xor(result, b)
  return result

def rotate_left(b, places):
  """Rotate a byte array to the left by a given number of places."""
  places = places % len(b)
  return b[places:] + b[:places]

def rotate_right(b, places):
  """Rotate a byte array to the right by a given number of places."""
  return rotate_left(b, -places)

def shift_left(b, places):
  """Shift a byte array to the left by a given number of places."""
  n = bytes_to_int(b)
  shifted = (n << places) & ((1 << (len(b) * 8)) - 1)
  return int_to_bytes(shifted, len(b))

def shift_right(b, places):
  """Shift a byte array to the right by a given number of places."""
  n = bytes_to_int(b)
  shifted = n >> places
  return int_to_bytes(shifted, len(b))

def key_length_in_bits(k):
  """Return the length of a byte array in bits."""
  return len(k) * 8

def hexstring_to_bytes(h):
  """Convert a hexstring to a byte array."""
  return bytes.fromhex(h)

def bytes_to_hexstring(b):
  """Convert a byte array to a hexstring."""
  return b.hex()


# --------------------------------------------------------------
# Format conversions

def quadSplit(b128):
    """Take a 128-bit byte array and return it as a list of 4 32-bit
    byte arrays, least significant byte array first."""
    
    if len(b128) != 16:
        raise ValueError("must be 16 bytes long, not " + str(len(b128)))
    
    result = []
    for i in range(4):
        result.append(b128[(i*4):(i+1)*4])
    return result


def quadJoin(l4x32):
    """Take a list of 4 32-bit byte arrays and return it as a single 128-bit
    byte array obtained by concatenating the internal ones."""

    if len(l4x32) != 4:
        raise ValueError("need a list of 4 byte arrays, not " + str(len(l4x32)))

    return l4x32[0] + l4x32[1] + l4x32[2] + l4x32[3]

# --------------------------------------------------------------
# Constants
phi = 0x9e3779b9
r = 32
# --------------------------------------------------------------
# Data tables


# Each element of this list corresponds to one S-box. Each S-box in turn is
# a list of 16 integers in the range 0..15, without repetitions. Having the
# value v (say, 14) in position p (say, 0) means that if the input to that
# S-box is the pattern p (0, or 0x0) then the output will be the pattern v
# (14, or 0xe).
SBoxDecimalTable = [
	[ 3, 8,15, 1,10, 6, 5,11,14,13, 4, 2, 7, 0, 9,12 ], # S0
	[15,12, 2, 7, 9, 0, 5,10, 1,11,14, 8, 6,13, 3, 4 ], # S1
	[ 8, 6, 7, 9, 3,12,10,15,13, 1,14, 4, 0,11, 5, 2 ], # S2
	[ 0,15,11, 8,12, 9, 6, 3,13, 1, 2, 4,10, 7, 5,14 ], # S3
	[ 1,15, 8, 3,12, 0,11, 6, 2, 5, 4,10, 9,14, 7,13 ], # S4
	[15, 5, 2,11, 4,10, 9,12, 0, 3,14, 8,13, 6, 7, 1 ], # S5
	[ 7, 2,12, 5, 8, 4, 6,11,14, 9, 1,15,13, 3,10, 0 ], # S6
	[ 1,13,15, 0,14, 8, 2,11, 7, 4,12,10, 9, 3, 5, 6 ], # S7
    ] 
# NB: in serpent-0, this was a list of 32 sublists (for the 32 different
# S-boxes derived from DES). In the final version of Serpent only 8 S-boxes
# are used, with each one being reused 4 times.


# Make another version of this table as a list of dictionaries: one
# dictionary per S-box, where the value of the entry indexed by i tells you
# the output configuration when the input is i, with both the index and the
# value being bitstrings.  Make also the inverse: another list of
# dictionaries, one per S-box, where each dictionary gets the output of the
# S-box as the key and gives you the input, with both values being 4-bit
# bitstrings.
SBoxBitstring = []
SBoxBitstringInverse = []
for line in SBoxDecimalTable:
    dict = {}
    inverseDict = {}
    for i in range(len(line)):
        index = int_to_bytes(i, 1)
        value = int_to_bytes(line[i], 1)
        dict[index] = value
        inverseDict[value] = index
    SBoxBitstring.append(dict)
    SBoxBitstringInverse.append(inverseDict)

# The Initial and Final permutations are each represented by one list
# containing the integers in 0..127 without repetitions.  Having value v
# (say, 32) at position p (say, 1) means that the output bit at position p
# (1) comes from the input bit at position v (32).
IPTable = [
    0, 32, 64, 96, 1, 33, 65, 97, 2, 34, 66, 98, 3, 35, 67, 99,
    4, 36, 68, 100, 5, 37, 69, 101, 6, 38, 70, 102, 7, 39, 71, 103,
    8, 40, 72, 104, 9, 41, 73, 105, 10, 42, 74, 106, 11, 43, 75, 107,
    12, 44, 76, 108, 13, 45, 77, 109, 14, 46, 78, 110, 15, 47, 79, 111,
    16, 48, 80, 112, 17, 49, 81, 113, 18, 50, 82, 114, 19, 51, 83, 115,
    20, 52, 84, 116, 21, 53, 85, 117, 22, 54, 86, 118, 23, 55, 87, 119,
    24, 56, 88, 120, 25, 57, 89, 121, 26, 58, 90, 122, 27, 59, 91, 123,
    28, 60, 92, 124, 29, 61, 93, 125, 30, 62, 94, 126, 31, 63, 95, 127,
    ]
FPTable = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60,
    64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124,
    1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61,
    65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125,
    2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62,
    66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126,
    3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63,
    67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127,
 ]

    
# The Linear Transformation is represented as a list of 128 lists, one for
# each output bit. Each one of the 128 lists is composed of a variable
# number of integers in 0..127 specifying the positions of the input bits
# that must be XORed together (say, 72, 144 and 125) to yield the output
# bit corresponding to the position of that list (say, 1).
LTTable = [
    [16, 52, 56, 70, 83, 94, 105],
    [72, 114, 125],
    [2, 9, 15, 30, 76, 84, 126],
    [36, 90, 103],
    [20, 56, 60, 74, 87, 98, 109],
    [1, 76, 118],
    [2, 6, 13, 19, 34, 80, 88],
    [40, 94, 107],
    [24, 60, 64, 78, 91, 102, 113],
    [5, 80, 122],
    [6, 10, 17, 23, 38, 84, 92],
    [44, 98, 111],
    [28, 64, 68, 82, 95, 106, 117],
    [9, 84, 126],
    [10, 14, 21, 27, 42, 88, 96],
    [48, 102, 115],
    [32, 68, 72, 86, 99, 110, 121],
    [2, 13, 88],
    [14, 18, 25, 31, 46, 92, 100],
    [52, 106, 119],
    [36, 72, 76, 90, 103, 114, 125],
    [6, 17, 92],
    [18, 22, 29, 35, 50, 96, 104],
    [56, 110, 123],
    [1, 40, 76, 80, 94, 107, 118],
    [10, 21, 96],
    [22, 26, 33, 39, 54, 100, 108],
    [60, 114, 127],
    [5, 44, 80, 84, 98, 111, 122],
    [14, 25, 100],
    [26, 30, 37, 43, 58, 104, 112],
    [3, 118],
    [9, 48, 84, 88, 102, 115, 126],
    [18, 29, 104],
    [30, 34, 41, 47, 62, 108, 116],
    [7, 122],
    [2, 13, 52, 88, 92, 106, 119],
    [22, 33, 108],
    [34, 38, 45, 51, 66, 112, 120],
    [11, 126],
    [6, 17, 56, 92, 96, 110, 123],
    [26, 37, 112],
    [38, 42, 49, 55, 70, 116, 124],
    [2, 15, 76],
    [10, 21, 60, 96, 100, 114, 127],
    [30, 41, 116],
    [0, 42, 46, 53, 59, 74, 120],
    [6, 19, 80],
    [3, 14, 25, 100, 104, 118],
    [34, 45, 120],
    [4, 46, 50, 57, 63, 78, 124],
    [10, 23, 84],
    [7, 18, 29, 104, 108, 122],
    [38, 49, 124],
    [0, 8, 50, 54, 61, 67, 82],
    [14, 27, 88],
    [11, 22, 33, 108, 112, 126],
    [0, 42, 53],
    [4, 12, 54, 58, 65, 71, 86],
    [18, 31, 92],
    [2, 15, 26, 37, 76, 112, 116],
    [4, 46, 57],
    [8, 16, 58, 62, 69, 75, 90],
    [22, 35, 96],
    [6, 19, 30, 41, 80, 116, 120],
    [8, 50, 61],
    [12, 20, 62, 66, 73, 79, 94],
    [26, 39, 100],
    [10, 23, 34, 45, 84, 120, 124],
    [12, 54, 65],
    [16, 24, 66, 70, 77, 83, 98],
    [30, 43, 104],
    [0, 14, 27, 38, 49, 88, 124],
    [16, 58, 69],
    [20, 28, 70, 74, 81, 87, 102],
    [34, 47, 108],
    [0, 4, 18, 31, 42, 53, 92],
    [20, 62, 73],
    [24, 32, 74, 78, 85, 91, 106],
    [38, 51, 112],
    [4, 8, 22, 35, 46, 57, 96],
    [24, 66, 77],
    [28, 36, 78, 82, 89, 95, 110],
    [42, 55, 116],
    [8, 12, 26, 39, 50, 61, 100],
    [28, 70, 81],
    [32, 40, 82, 86, 93, 99, 114],
    [46, 59, 120],
    [12, 16, 30, 43, 54, 65, 104],
    [32, 74, 85],
    [36, 90, 103, 118],
    [50, 63, 124],
    [16, 20, 34, 47, 58, 69, 108],
    [36, 78, 89],
    [40, 94, 107, 122],
    [0, 54, 67],
    [20, 24, 38, 51, 62, 73, 112],
    [40, 82, 93],
    [44, 98, 111, 126],
    [4, 58, 71],
    [24, 28, 42, 55, 66, 77, 116],
    [44, 86, 97],
    [2, 48, 102, 115],
    [8, 62, 75],
    [28, 32, 46, 59, 70, 81, 120],
    [48, 90, 101],
    [6, 52, 106, 119],
    [12, 66, 79],
    [32, 36, 50, 63, 74, 85, 124],
    [52, 94, 105],
    [10, 56, 110, 123],
    [16, 70, 83],
    [0, 36, 40, 54, 67, 78, 89],
    [56, 98, 109],
    [14, 60, 114, 127],
    [20, 74, 87],
    [4, 40, 44, 58, 71, 82, 93],
    [60, 102, 113],
    [3, 18, 72, 114, 118, 125],
    [24, 78, 91],
    [8, 44, 48, 62, 75, 86, 97],
    [64, 106, 117],
    [1, 7, 22, 76, 118, 122],
    [28, 82, 95],
    [12, 48, 52, 66, 79, 90, 101],
    [68, 110, 121],
    [5, 11, 26, 80, 122, 126],
    [32, 86, 99],
    ]

# The following table is necessary for the non-bitslice decryption.
LTTableInverse = [
    [53, 55, 72],
    [1, 5, 20, 90],
    [15, 102],
    [3, 31, 90],
    [57, 59, 76],
    [5, 9, 24, 94],
    [19, 106],
    [7, 35, 94],
    [61, 63, 80],
    [9, 13, 28, 98],
    [23, 110],
    [11, 39, 98],
    [65, 67, 84],
    [13, 17, 32, 102],
    [27, 114],
    [1, 3, 15, 20, 43, 102],
    [69, 71, 88],
    [17, 21, 36, 106],
    [1, 31, 118],
    [5, 7, 19, 24, 47, 106],
    [73, 75, 92],
    [21, 25, 40, 110],
    [5, 35, 122],
    [9, 11, 23, 28, 51, 110],
    [77, 79, 96],
    [25, 29, 44, 114],
    [9, 39, 126],
    [13, 15, 27, 32, 55, 114],
    [81, 83, 100],
    [1, 29, 33, 48, 118],
    [2, 13, 43],
    [1, 17, 19, 31, 36, 59, 118],
    [85, 87, 104],
    [5, 33, 37, 52, 122],
    [6, 17, 47],
    [5, 21, 23, 35, 40, 63, 122],
    [89, 91, 108],
    [9, 37, 41, 56, 126],
    [10, 21, 51],
    [9, 25, 27, 39, 44, 67, 126],
    [93, 95, 112],
    [2, 13, 41, 45, 60],
    [14, 25, 55],
    [2, 13, 29, 31, 43, 48, 71],
    [97, 99, 116],
    [6, 17, 45, 49, 64],
    [18, 29, 59],
    [6, 17, 33, 35, 47, 52, 75],
    [101, 103, 120],
    [10, 21, 49, 53, 68],
    [22, 33, 63],
    [10, 21, 37, 39, 51, 56, 79],
    [105, 107, 124],
    [14, 25, 53, 57, 72],
    [26, 37, 67],
    [14, 25, 41, 43, 55, 60, 83],
    [0, 109, 111],
    [18, 29, 57, 61, 76],
    [30, 41, 71],
    [18, 29, 45, 47, 59, 64, 87],
    [4, 113, 115],
    [22, 33, 61, 65, 80],
    [34, 45, 75],
    [22, 33, 49, 51, 63, 68, 91],
    [8, 117, 119],
    [26, 37, 65, 69, 84],
    [38, 49, 79],
    [26, 37, 53, 55, 67, 72, 95],
    [12, 121, 123],
    [30, 41, 69, 73, 88],
    [42, 53, 83],
    [30, 41, 57, 59, 71, 76, 99],
    [16, 125, 127],
    [34, 45, 73, 77, 92],
    [46, 57, 87],
    [34, 45, 61, 63, 75, 80, 103],
    [1, 3, 20],
    [38, 49, 77, 81, 96],
    [50, 61, 91],
    [38, 49, 65, 67, 79, 84, 107],
    [5, 7, 24],
    [42, 53, 81, 85, 100],
    [54, 65, 95],
    [42, 53, 69, 71, 83, 88, 111],
    [9, 11, 28],
    [46, 57, 85, 89, 104],
    [58, 69, 99],
    [46, 57, 73, 75, 87, 92, 115],
    [13, 15, 32],
    [50, 61, 89, 93, 108],
    [62, 73, 103],
    [50, 61, 77, 79, 91, 96, 119],
    [17, 19, 36],
    [54, 65, 93, 97, 112],
    [66, 77, 107],
    [54, 65, 81, 83, 95, 100, 123],
    [21, 23, 40],
    [58, 69, 97, 101, 116],
    [70, 81, 111],
    [58, 69, 85, 87, 99, 104, 127],
    [25, 27, 44],
    [62, 73, 101, 105, 120],
    [74, 85, 115],
    [3, 62, 73, 89, 91, 103, 108],
    [29, 31, 48],
    [66, 77, 105, 109, 124],
    [78, 89, 119],
    [7, 66, 77, 93, 95, 107, 112],
    [33, 35, 52],
    [0, 70, 81, 109, 113],
    [82, 93, 123],
    [11, 70, 81, 97, 99, 111, 116],
    [37, 39, 56],
    [4, 74, 85, 113, 117],
    [86, 97, 127],
    [15, 74, 85, 101, 103, 115, 120],
    [41, 43, 60],
    [8, 78, 89, 117, 121],
    [3, 90],
    [19, 78, 89, 105, 107, 119, 124],
    [45, 47, 64],
    [12, 82, 93, 121, 125],
    [7, 94],
    [0, 23, 82, 93, 109, 111, 123],
    [49, 51, 68],
    [1, 16, 86, 97, 125],
    [11, 98],
    [4, 27, 86, 97, 113, 115, 127],
]
