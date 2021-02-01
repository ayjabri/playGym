from numpy import zeros, uint8


cpdef int bits_to_int(list bits):
    cdef int res = 0
    cdef int b
    for b in bits:
        res *= 2
        res += b
    return res

cpdef list int_to_bits(int num,int bits):
    cdef list res = []
    for _ in range(bits):
        res.append(num % 2)
        num //= 2
    return res[::-1]

cpdef int encode_lists(field_lists,int rows=6,int bits_in_len=3):
    """
    Encode a list representation into binary number
    """
    cdef list bits = []
    cdef list len_bits = []
    cdef list col
    cdef int free_len
    for col in field_lists:
        bits.extend(col)
        free_len = rows-len(col)
        bits.extend([0] * free_len)
        len_bits.extend(int_to_bits(free_len, bits=bits_in_len))
    bits.extend(len_bits)
    return bits_to_int(bits)

cpdef list decode_binary(int state_int,int cols=7,int rows=6,int bits_in_len=3):
    """
    Decode binary representation into the list view
    :param state_int: integer representing the field
    :return: list of GAME_COLS lists
    """
    cdef list res = []
    cdef list bits
    cdef list len_bits
    cdef int col
    cdef list vals 
    bits = int_to_bits(state_int, bits=cols*rows + cols*bits_in_len)
    len_bits = bits[cols*rows:]
    for col in range(cols):
        vals = bits[col*rows:(col+1)*rows]
        lens = bits_to_int(len_bits[col*bits_in_len:(col+1)*bits_in_len])
        if lens > 0:
            vals = vals[:-lens]
        res.append(vals)
    return res


cpdef binary_to_array(int state_int, int cols=7,int rows=6,int bits_in_len=3):
    cdef list len_bits, vals
    cdef int space, col, lens, i, cell
    space = cols*rows
    rep = zeros((2, cols, rows), dtype=uint8)
    bits = int_to_bits(
        state_int, bits=space+cols*bits_in_len)
    len_bits = bits[space:]
    for col in range(cols):
        vals = bits[col*rows:(col+1)*rows]
        lens = bits_to_int(
            len_bits[col*bits_in_len:(col+1)*bits_in_len])
        if lens > 0:
            vals = vals[:-lens]
            for i, cell in enumerate(vals):
                if cell == 0:
                    rep[0, col, i] += 1
                else:
                    rep[1, col, i] += 1
    return rep