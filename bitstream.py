import numpy as np

class BitstreamReader:
    """
    Utility to read bits from a packed uint8 byte stream.
    Used for parsing entropy-coded or bit-packed data streams.
    """
    def __init__(self, data: np.ndarray):
        """ 
        data: uint8 numpy array
        """
        self.data = np.asarray(data, dtype=np.uint8).flatten()
        self.byte_pos = 0
        self.bit_pos = 0
        self.total_bytes = len(self.data)
        
    def read_bit(self):
        """Reads a single bit and returns 0 or 1"""
        if self.byte_pos >= self.total_bytes:
            raise EOFError("Reached end of bitstream")
            
        byte_val = self.data[self.byte_pos]
        # Read from most significant to least significant bit
        bit_val = (byte_val >> (7 - self.bit_pos)) & 1
        
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.bit_pos = 0
            self.byte_pos += 1
            
        return bit_val

    def read_bits(self, num_bits):
        """Reads num_bits as an integer"""
        val = 0
        for _ in range(num_bits):
            val = (val << 1) | self.read_bit()
        return val

# TODO: Depending on how the dahuffman/entropy encoder is integrated in the NPZ, 
# we might need to rely on the actual `dahuffman` package to decode the byte streams natively.
# This bitstream utility serves as a manual fallback if we implement custom bit-packing.

def decode_huffman_stub(compressed_bytes, huffman_table, num_elements):
    """
    Mock function. If actual dahuffman is not used or available, 
    we need to implement tree traversal here.
    """
    # In a full implementation, you'd traverse the huffman table tree 
    # reading from BitstreamReader.
    print("TODO: Implement true huffman decoding. Returning mock zero data.")
    return np.zeros(num_elements, dtype=np.float32)
