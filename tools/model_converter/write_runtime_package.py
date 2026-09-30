"""Convert a user-provided DLSS NR DLL into the pinned NRWGT001 package.

The DLL is parsed as PE data. It is never imported or executed.
"""
import argparse
import hashlib
import json
import os
import struct
import uuid
from pathlib import Path

import numpy as np

from model_weight_provider import ModelWeightProvider


HEADER_SIZE = 128
ENTRY_SIZE = 128
ALIGNMENT = 64
SOURCE_SHA256 = 'E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E'
PACKAGE_SHA256 = 'A7E6EE38172A81E12D613FA9A2F57E32AA1944908E56CD2A33E1F6C94369E3CB'
PACKAGE_BYTES = 291595458


def iter_tensors(provider):
    for block in range(71):
        weights = provider(block)
        values = {}
        for source_role in (key for key in weights if key != 'family'):
            value = weights[source_role]
            role = source_role
            if source_role == 'scale_fp32':
                role = 'effective_attention_scale'
                value = np.asarray(value, np.float32) * np.float32(np.sqrt(32))
            values[role] = np.ascontiguousarray(value, dtype='<f2')
        if block == 70:
            scalar = provider.records['block70.layer0.blend_scale']
            start = scalar['sourceOffset']
            raw = provider.data[start:start + scalar['byteLength']]
            if raw != bytes.fromhex('eb39'):
                raise ValueError('Unexpected temporal blend scalar')
            values['temporal_blend_scale'] = np.frombuffer(raw, '<f2')
        for role in sorted(values):
            yield block, role, values[role]


def metadata(provider):
    result = []
    for block, role, array in iter_tensors(provider):
        encoded = role.encode('ascii')
        if not 0 < len(encoded) < 32 or array.ndim > 6 or not np.isfinite(array).all():
            raise ValueError(f'Invalid tensor {block}:{role}')
        result.append({
            'block': block,
            'role': role,
            'shape': list(array.shape),
            'byteLength': array.nbytes,
            'sha256': hashlib.sha256(array.tobytes()).digest(),
        })
    if result != sorted(result, key=lambda value: (value['block'], value['role'])):
        raise ValueError('Tensor order is not canonical')
    if len({(value['block'], value['role']) for value in result}) != len(result):
        raise ValueError('Duplicate tensor key')
    offset = HEADER_SIZE + ENTRY_SIZE * len(result)
    for value in result:
        offset = (offset + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        value['packageOffset'] = offset
        offset += value['byteLength']
    return result, offset


def entry_bytes(value):
    shape = value['shape'] + [0] * (6 - len(value['shape']))
    role = value['role'].encode('ascii') + b'\0'
    role += bytes(32 - len(role))
    return struct.pack(
        '<4I6I32sQQ32s8s',
        value['block'], 1, len(value['shape']), 0, *shape, role,
        value['packageOffset'], value['byteLength'], value['sha256'], bytes(8),
    )


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.digest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='your original nvngx_dlssnr.dll')
    parser.add_argument('--output', required=True, type=Path, help='destination .nrwgt file')
    parser.add_argument('--report', type=Path, help='optional JSON receipt (no weight data)')
    args = parser.parse_args()
    output = args.output
    if args.source.resolve() == output.resolve():
        raise ValueError('source and output must differ')
    output.parent.mkdir(parents=True, exist_ok=True)
    # A short sibling avoids Windows MAX_PATH surprises in deeply nested games.
    temporary = output.parent / ('nr-' + uuid.uuid4().hex + '.tmp')

    provider = ModelWeightProvider(args.source, SOURCE_SHA256)
    entries, file_size = metadata(provider)
    table = b''.join(entry_bytes(value) for value in entries)
    table_hash = hashlib.sha256(table).digest()
    payload_hash = hashlib.sha256()
    source_hash = bytes.fromhex(provider.selection['sha256'])

    with temporary.open('xb') as stream:
        stream.write(bytes(HEADER_SIZE))
        stream.write(table)
        current = stream.tell()
        tensors = iter(iter_tensors(provider))
        for expected in entries:
            block, role, array = next(tensors)
            if (block, role) != (expected['block'], expected['role']):
                raise ValueError('Second tensor pass changed order')
            padding = expected['packageOffset'] - current
            if padding < 0:
                raise ValueError('Package offset regression')
            zeros = bytes(padding)
            stream.write(zeros)
            payload_hash.update(zeros)
            raw = array.tobytes()
            if len(raw) != expected['byteLength'] or hashlib.sha256(raw).digest() != expected['sha256']:
                raise ValueError(f'Second tensor pass changed {block}:{role}')
            stream.write(raw)
            payload_hash.update(raw)
            current = stream.tell()
        try:
            next(tensors)
            raise ValueError('Second tensor pass has additional data')
        except StopIteration:
            pass
        if current != file_size:
            raise ValueError('Package size calculation mismatch')
        header = struct.pack(
            '<8sIIQII32s32s32s', b'NRWGT001', 1, 1, file_size,
            len(entries), ENTRY_SIZE, source_hash, table_hash, payload_hash.digest(),
        )
        if len(header) != HEADER_SIZE:
            raise ValueError('Header layout mismatch')
        stream.seek(0)
        stream.write(header)
        stream.flush()
        os.fsync(stream.fileno())

    package_hash = file_digest(temporary)
    if file_size != PACKAGE_BYTES or package_hash.hex().upper() != PACKAGE_SHA256:
        temporary.unlink(missing_ok=True)
        raise ValueError('converted package does not match the runtime contract')
    temporary.replace(output)
    report = {
        'contract': 'runtime-weight-package-build-v1',
        'format': 'NRWGT001',
        'path': str(output.resolve()),
        'byteLength': file_size,
        'sha256': package_hash.hex().upper(),
        'sourceSha256': source_hash.hex().upper(),
        'tableSha256': table_hash.hex().upper(),
        'payloadSha256': payload_hash.hexdigest().upper(),
        'tensorCount': len(entries),
        'tensorBytes': sum(value['byteLength'] for value in entries),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('PASS runtime package', file_size, package_hash.hex().upper())


if __name__ == '__main__':
    main()
