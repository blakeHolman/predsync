#!/usr/bin/env python3
# scripts/residuals.py

import difflib

# Define OP codes (COPY, LIT)
OP_COPY = 0x00  # COPY(off, len): copy bytes from pred[off:off+len]
OP_LIT  = 0x01  # LIT(len, bytes): append literal bytes

def _to_bytes(s):
    """
    Turns string into bytes
    
    Parameters:
        s (string): string of data
    
    Returns:
        bytes: byte representation of the string
    """
    return s.encode("utf-8")


def _enc_varint(n):
    """
    Given an int n, determines how many bytes are needed to represent it
    
    Parameters:
        n (int): location or length

    Returns:
        bytes: returns an array of bytes representing n in least amount of bytes
    """
    if n < 0:
        raise ValueError("varint must be non-negative")
    out = bytearray()
    while n >= 0x80:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    out.append(n & 0x7F)
    return bytes(out)


def _dec_varint(buf, k):
    """
    Decode an unsigned LEB128 varint from buf starting at index k.
    
    Parameters:
        buf (bytes): buffer of bytes
        k (int): start point

    Returns:
        (val, next_k)
    """
    shift = 0
    val = 0
    while True:
        if k >= len(buf):
            raise ValueError("truncated varint")
        b = buf[k]
        k += 1
        val |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return val, k
        shift += 7
        if shift > 63:
            raise ValueError("varint too large")


def get_residual(new, pred):
    """
    Given a new and predicted string, determine residual of COPY and LIT
    
    Parameters:
        new (string): actual string
        pred (string): predicted string

    Returns:
        bytes: returns a list of bytes containing the OPs to rebuild new based on residuals.
    """
    new_b = _to_bytes(new)
    pred_b = _to_bytes(pred)

    sm = difflib.SequenceMatcher(a=pred_b, b=new_b, autojunk=False)

    r = bytearray()

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            # Reuse bytes from pred
            length = i2 - i1
            if length:
                r.append(OP_COPY)
                r += _enc_varint(i1)
                r += _enc_varint(length)

        elif tag in ("replace", "insert"):
            # Send literal bytes from new
            lit = new_b[j1:j2]
            if lit:
                r.append(OP_LIT)
                r += _enc_varint(len(lit))
                r += lit

        elif tag == "delete":
            # Bytes present in pred but not in new => emit nothing
            continue

        else:
            raise RuntimeError(f"Unexpected opcode tag: {tag}")

    return bytes(r)



def apply_residual(pred, r):
    """
    Given a predicted string and residual op-codes, rebuild new
    
    Parameters:
        pred (string): predicted string
        r (bytearray): represents op-codes

    Returns:
        new: new string built from pred and r.
    """
    pred_b = _to_bytes(pred)
    out = bytearray()
    k = 0
    while k < len(r):
        op = r[k]
        k += 1
        if op == OP_COPY:
            off, k = _dec_varint(r, k)
            ln,  k = _dec_varint(r, k)
            out += pred_b[off:off + ln]
        elif op == OP_LIT:
            ln, k = _dec_varint(r, k)
            out += r[k:k + ln]
            k += ln
        else:
            raise ValueError(f"unknown op {op}")
    return bytes(out)