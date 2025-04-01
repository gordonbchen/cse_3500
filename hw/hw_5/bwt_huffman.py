import os
import sys
import marshal
import itertools
import argparse
from operator import itemgetter
from functools import partial
from collections import Counter
from heapq import heappush, heappop, heapify

try:
    import cPickle as pickle
except:
    import pickle

termchar = 17 # you can assume the byte 17 does not appear in the input file

# This takes a sequence of bytes over which you can iterate, msg, 
# and returns a tuple (enc,\ ring) in which enc is the ASCII representation of the 
# Huffman-encoded message (e.g. "1001011") and ring is your ``decoder ring'' needed 
# to decompress that message.
def encode(msg: bytes) -> tuple[str, dict[str, int]]:
    counts = Counter(msg)
    freq_heap = [(v, (k,)) for (k, v) in counts.items()]
    heapify(freq_heap)

    codes = {k: "" for k in counts.keys()}
    while len(freq_heap) > 1:
        left_count, left = heappop(freq_heap)
        right_count, right = heappop(freq_heap)
        
        for k in left:
            codes[k] = "0" + codes[k]
        for k in right:
            codes[k] = "1" + codes[k]

        count = left_count + right_count
        combined = left + right
        heappush(freq_heap, (count, combined))

    encoded = "".join(codes[i] for i in msg)
    # NOTE: decoderRing is the inverse of codes, maps from codes to msg ints.
    decoderRing = {v: k for (k, v) in codes.items()}
    return (encoded, decoderRing)

# This takes a string, cmsg, which must contain only 0s and 1s, and your 
# representation of the ``decoder ring'' ring, and returns a bytearray msg which 
# is the decompressed message. 
def decode(cmsg: str, decoderRing: dict[str, int]) -> bytearray:
    byteMsg = bytearray()
    curr_code = ""

    for i in cmsg:
        curr_code += i
        if curr_code in decoderRing:
            byteMsg.append(decoderRing[curr_code])
            curr_code = ""

    return byteMsg

# This takes a sequence of bytes over which you can iterate, msg, and returns a tuple (compressed, ring) 
# in which compressed is a bytearray (containing the Huffman-coded message in binary, 
# and ring is again the ``decoder ring'' needed to decompress the message.
def compress(msg: bytes, useBWT: bool) -> tuple[bytearray, dict[str, int]]:
    if useBWT:
        msg = bwt(msg)
        msg = mtf(msg)

    encoded, decoderRing = encode(msg)
    padding = (8 - (len(encoded) % 8)) % 8
    padded = encoded + ("0" * padding)
    
    # Compressed: 1 byte of # padding bits, message, padding bits.
    compressed = bytearray([padding] + [int(padded[i:i+8], base=2) for i in range(0, len(padded), 8)])
    return compressed, decoderRing

# This takes a sequence of bytes over which you can iterate containing the Huffman-coded message, and the 
# decoder ring needed to decompress it.  It returns the bytearray which is the decompressed message. 
def decompress(msg: bytes, decoderRing: dict[str, int], useBWT: bool) -> bytearray:
    padding = msg[0]
    binary = "".join(format(byte, "08b") for byte in bytearray(msg[1:]))
    decompressedMsg = decode(binary[:len(binary)-padding], decoderRing)

    # before you return, you must invert the move-to-front and BWT if applicable
    # here, decompressed message should be the return value from decode()
    if useBWT:
        decompressedMsg = imtf(decompressedMsg)
        decompressedMsg = ibwt(decompressedMsg)

    return decompressedMsg

# memory efficient iBWT
def ibwt(msg: bytearray) -> bytearray:
    first = sorted(msg)
    first_ranks = {}
    for (i, c) in enumerate(first):
        if c not in first_ranks:
            first_ranks[c] = []
        first_ranks[c].append(i)

    counts = {}
    ranks = []
    for c in msg:
        rank = counts.get(c, 0)
        ranks.append(rank)
        counts[c] = rank + 1

    recon = bytearray()
    i = first.index(termchar)
    for _ in range(len(msg)):
        recon.append(first[i])
        i = first_ranks[msg[i]][ranks[i]]
    recon.reverse()
    return recon[:-1]  # remove term char.

# Burrows-Wheeler Transform fncs
def radix_sort(values, key, step=0):
    sortedvals = []
    radix_stack = []
    radix_stack.append((values, key, step))

    while len(radix_stack) > 0:
        values, key, step = radix_stack.pop()
        if len(values) < 2:
            for value in values:
                sortedvals.append(value)
            continue

        bins = {}
        for value in values:
            bins.setdefault(key(value, step), []).append(value)

        for k in sorted(bins.keys()):
            radix_stack.append((bins[k], key, step + 1))
    return sortedvals
            
# memory efficient BWT
def bwt(msg):
    def bw_key(text, value, step):
        return text[(value + step) % len(text)]

    msg = msg + termchar.to_bytes(1, byteorder='big')

    bwtM = bytearray()

    rs = radix_sort(range(len(msg)), partial(bw_key, msg))
    for i in rs:
        bwtM.append(msg[i - 1])

    return bwtM[::-1]

# move-to-front encoding fncs
def mtf(msg):
    # Initialise the list of characters (i.e. the dictionary)
    dictionary = bytearray(range(256))
    
    # Transformation
    compressed_text = bytearray()
    rank = 0

    # read in each character
    for c in msg:
        rank = dictionary.index(c) # find the rank of the character in the dictionary
        compressed_text.append(rank) # update the encoded text
        
        # update the dictionary
        dictionary.pop(rank)
        dictionary.insert(0, c)

    #dictionary.sort() # sort dictionary
    return compressed_text # Return the encoded text as well as the dictionary

# inverse move-to-front
def imtf(compressed_msg):
    compressed_text = compressed_msg
    dictionary = bytearray(range(256))

    decompressed_img = bytearray()

    # read in each character of the encoded text
    for i in compressed_text:
        # read the rank of the character from dictionary
        decompressed_img.append(dictionary[i])
        
        # update dictionary
        e = dictionary.pop(i)
        dictionary.insert(0, e)
        
    return decompressed_img # Return original string

if __name__=='__main__':

    # argparse is an excellent library for parsing arguments to a python program
    parser = argparse.ArgumentParser(description='Dianoga (Star Wars reference) compresses '
                                                 'binary and plain text files using the Burrows-Wheeler transform, '
                                                 'move-to-front coding, and Huffman coding.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-c', action='store_true', help='Compresses a stream of bytes (e.g. file) into a bytes.')
    group.add_argument('-d', action='store_true', help='Decompresses a compressed file back into the original input')
    group.add_argument('-v', action='store_true', help='Encodes a stream of bytes (e.g. file) into a binary string'
                                                       ' using Huffman encoding.')
    group.add_argument('-w', action='store_true', help='Decodes a Huffman encoded binary string into bytes.')
    parser.add_argument('-i', '--input', help='Input file path', required=True)
    parser.add_argument('-o', '--output', help='Output file path', required=True)
    parser.add_argument('-b', '--binary', help='Use this option if the file is binary and therefore '
                                               'do not want to use the BWT.', action='store_true')

    args = parser.parse_args()

    compressing = args.c
    decompressing = args.d
    encoding = args.v
    decoding = args.w


    infile = args.input
    outfile = args.output
    useBWT = not args.binary

    assert os.path.exists(infile)

    if compressing or encoding:
        fp = open(infile, 'rb')
        sinput = fp.read()
        fp.close()
        if compressing:
            msg, tree = compress(sinput,useBWT)
            fcompressed = open(outfile, 'wb')
            marshal.dump((pickle.dumps(tree), msg), fcompressed)
            fcompressed.close()
        else:
            msg, tree = encode(sinput)
            print(msg)
            fcompressed = open(outfile, 'wb')
            marshal.dump((pickle.dumps(tree), msg), fcompressed)
            fcompressed.close()
    else:
        fp = open(infile, 'rb')
        pck, msg = marshal.load(fp)
        tree = pickle.loads(pck)
        fp.close()
        if decompressing:
            sinput = decompress(msg, tree, useBWT)
        else:
            sinput = decode(msg, tree)
            print(sinput)
        fp = open(outfile, 'wb')
        fp.write(sinput)
        fp.close()