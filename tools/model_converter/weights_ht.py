"""Read the observed WEIGHTS_HT container; preserve opaque payload semantics.

Format evidence: local 310.8 DLL deserializer at RVA 0x44CF0 and byte traversal.
Cross-reference: MLX-DLSS 0ca2deab, extract_dlssnr_weights.py (Apache-2.0).
No vendor code is loaded. This module uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

MAX_FILE = 256 * 1024 * 1024
MAX_RECORDS = 4096


def digest(data):
    return hashlib.sha256(data).hexdigest().upper()


class Reader:
    def __init__(self, data):
        self.data = memoryview(data)

    def span(self, offset, size):
        if offset < 0 or size < 0 or offset > len(self.data) - size:
            raise ValueError(f'out-of-bounds range: offset={offset}, size={size}')
        return self.data[offset:offset + size]

    def integer(self, offset, size):
        return int.from_bytes(self.span(offset, size), 'little')


def file_offset(sections, rva, size):
    matches = []
    for va, raw, length in sections:
        relative = rva - va
        if 0 <= relative <= length and 0 <= size <= length - relative:
            matches.append(raw + relative)
    if len(matches) != 1:
        raise ValueError('RVA must have exactly one file-backed section mapping')
    return matches[0]


def locate_resource(data):
    """Strict PE32+ resource lookup for RCDATA/WEIGHTS_HT/one language."""
    r = Reader(data)
    if r.span(0, 2) != b'MZ':
        raise ValueError('missing MZ')
    pe = r.integer(0x3c, 4)
    if r.span(pe, 4) != b'PE\0\0' or r.integer(pe + 4, 2) != 0x8664:
        raise ValueError('expected x64 PE')
    count, optional_size = r.integer(pe + 6, 2), r.integer(pe + 20, 2)
    if not 1 <= count <= 96 or optional_size < 136:
        raise ValueError('unsupported section count or optional header')
    optional = pe + 24
    r.span(optional, optional_size)
    if r.integer(optional, 2) != 0x20b or r.integer(optional + 108, 4) < 3:
        raise ValueError('missing PE32+ resource directory')
    resource_rva = r.integer(optional + 128, 4)
    resource_size = r.integer(optional + 132, 4)
    if not resource_rva or resource_size < 16:
        raise ValueError('empty resource directory')
    sections = []
    for index in range(count):
        start = optional + optional_size + index * 40
        r.span(start, 40)
        va, size, raw = (r.integer(start + off, 4) for off in (12, 16, 20))
        r.span(raw, size)
        sections.append((va, raw, size))
    base = file_offset(sections, resource_rva, resource_size)
    tree = Reader(r.span(base, resource_size))
    visited = set()
    budget = 0

    def entries(relative):
        nonlocal budget
        if relative in visited:
            raise ValueError('cyclic/shared resource directory')
        visited.add(relative)
        tree.span(relative, 16)
        count = tree.integer(relative + 12, 2) + tree.integer(relative + 14, 2)
        budget += count
        if budget > MAX_RECORDS:
            raise ValueError('resource entry budget exceeded')
        tree.span(relative + 16, 8 * count)
        return [(tree.integer(relative + 16 + i * 8, 4),
                 tree.integer(relative + 20 + i * 8, 4)) for i in range(count)]

    def subdirectory(target):
        if not target & 0x80000000:
            raise ValueError('expected resource subdirectory')
        return target & 0x7fffffff

    types = [target for name, target in entries(0) if name == 10]
    if len(types) != 1:
        raise ValueError('expected one RCDATA entry')
    named = []
    for name, target in entries(subdirectory(types[0])):
        if name & 0x80000000:
            offset = name & 0x7fffffff
            length = tree.integer(offset, 2)
            if length > 256:
                raise ValueError('oversized resource name')
            text = bytes(tree.span(offset + 2, length * 2)).decode('utf-16le')
            if text == 'WEIGHTS_HT':
                named.append(target)
    if len(named) != 1:
        raise ValueError('expected one WEIGHTS_HT entry')
    languages = entries(subdirectory(named[0]))
    if len(languages) != 1:
        raise ValueError('expected one resource language')
    language, target = languages[0]
    if language & 0x80000000 or target & 0x80000000:
        raise ValueError('invalid resource leaf')
    tree.span(target, 16)
    rva, size = tree.integer(target, 4), tree.integer(target + 4, 4)
    if not 8 <= size <= MAX_FILE:
        raise ValueError('invalid resource size')
    offset = file_offset(sections, rva, size)
    r.span(offset, size)
    return {'path': ['RCDATA', 'WEIGHTS_HT', language], 'rva': rva,
            'fileOffset': offset, 'byteLength': size,
            'sha256': digest(r.span(offset, size))}


def parse_records(data, source_offset=0):
    r = Reader(data)
    if len(data) > MAX_FILE or r.integer(0, 8) != len(data):
        raise ValueError('container size mismatch')
    cursor, records, names = 8, [], set()
    while cursor < len(data):
        if len(records) >= MAX_RECORDS:
            raise ValueError('record budget exceeded')
        start = cursor
        length = r.integer(cursor, 8)
        if not 1 <= length <= 4096:
            raise ValueError('invalid name length')
        name = bytes(r.span(cursor + 8, length)).decode('utf-8')
        if name in names or '\0' in name:
            raise ValueError('duplicate or NUL-containing record name')
        names.add(name)
        cursor += 8 + length
        outer = r.integer(cursor, 8)
        blob_start = cursor + 8
        blob = Reader(r.span(blob_start, outer))
        if outer < 40 or blob.integer(0, 8) != outer:
            raise ValueError('inner/outer size mismatch')
        byte_length, device = blob.integer(8, 8), blob.integer(16, 4)
        if byte_length == 0 or device != 1:
            raise ValueError('unsupported empty payload/device code')
        payload = blob.span(20, byte_length)
        meta = 20 + byte_length
        metadata0, metadata1 = blob.integer(meta, 4), blob.integer(meta + 4, 4)
        rank = blob.integer(meta + 8, 8)
        if not 1 <= rank <= 16 or meta + 16 + rank * 4 != outer:
            raise ValueError('invalid rank or trailing metadata')
        dimensions = [blob.integer(meta + 16 + i * 4, 4) for i in range(rank)]
        if any(x == 0 for x in dimensions) or math.prod(dimensions) * 2 != byte_length:
            raise ValueError('unsupported storage dimensions/byte count')
        if metadata0 != 0 or metadata1 != 0:
            raise ValueError('unsupported metadata codes for pinned format')
        payload_offset = blob_start + 20
        records.append({
            'index': len(records), 'name': name, 'recordOffset': start,
            'recordEndExclusive': blob_start + outer, 'blobOffset': blob_start,
            'blobByteLength': outer, 'payloadOffset': payload_offset,
            'sourceOffset': source_offset + payload_offset, 'byteLength': byte_length,
            'sha256': digest(payload), 'deviceCode': device,
            'metadata0': metadata0, 'metadata1': metadata1,
            'serializedDimensions': dimensions, 'serializedElementBytes': 2,
            'logicalDtype': 'unknown', 'logicalShape': None, 'logicalLayout': None,
        })
        cursor = blob_start + outer
    if not records:
        raise ValueError('empty weight map')
    return records


def load_file(path, expected_sha256):
    """Read a user-selected DLL as data; never load or execute it."""
    path = Path(path)
    selection = {'path': str(path), 'sha256': expected_sha256}
    if path.stat().st_size > MAX_FILE:
        raise ValueError('model file exceeds inspection limit')
    data = path.read_bytes()
    if digest(data) != selection['sha256']:
        raise ValueError('unsupported source DLL (SHA-256 does not match this runtime)')
    resource = locate_resource(data)
    start = resource['fileOffset']
    payload = memoryview(data)[start:start + resource['byteLength']]
    records = parse_records(payload, start)
    return selection, data, resource, records
