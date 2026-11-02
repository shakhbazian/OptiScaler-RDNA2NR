"""Identity and converter regression checks; optional real package verification."""
import hashlib
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/model_converter'))
from model_profiles import PROFILES, PACKAGES, SOURCE_A, SOURCE_AB, select_profile, catalog_header
from weights_ht import locate_resource, parse_records


class Profiles(unittest.TestCase):
    def test_generated_runtime_catalog(self):
        header=Path(__file__).resolve().parents[2]/'rdna2_nr/include/nr_model_catalog.h'
        self.assertEqual(header.read_text(encoding='utf-8'),catalog_header())

    def test_truncated_pe_and_records(self):
        for raw in (b'', b'MZ', b'MZ'+bytes(1022)):
            with self.assertRaises(ValueError):
                locate_resource(raw, 'WEIGHTS_CC_SMALL_FP8')
        for raw in (b'', bytes(7), bytes(32)):
            with self.assertRaises(ValueError):
                parse_records(raw, 0)

    def test_source_model_compatibility(self):
        self.assertEqual(select_profile(SOURCE_A, 'original').graph, 1)
        self.assertEqual(select_profile(SOURCE_AB, 'optimized').graph, 2)
        with self.assertRaises(ValueError):
            select_profile(SOURCE_A, 'optimized')
        with self.assertRaises(ValueError):
            select_profile('00' * 32, 'original')

    def test_distinct_layouts_and_blend(self):
        a, b = PROFILES.values()
        self.assertEqual((a.blocks, b.blocks), (71, 31))
        self.assertEqual(len(set(b.templates)), b.blocks)
        self.assertEqual(b.templates[11:19], tuple(range(31, 39)))
        self.assertNotEqual(a.blend_hex, b.blend_hex)


def verify_packages(folder):
    headers = {}
    for name, source, model in [('regression-a', SOURCE_A, 'original'),
                                ('model-a', SOURCE_AB, 'original'),
                                ('model-b', SOURCE_AB, 'optimized')]:
        path = folder / (name + '.nrwgt')
        raw = path.read_bytes()
        assert (len(raw), hashlib.sha256(raw).hexdigest().upper()) == PACKAGES[source, model]
        header = struct.unpack('<8sIIQII32s32s32s', raw[:128])
        assert header[1] == PROFILES[model].graph
        assert header[6].hex().upper() == source
        headers[name] = header
    # Both source DLLs contain the very same A weights; only provenance changes.
    assert headers['regression-a'][7:] == headers['model-a'][7:]
    print('PASS all package identities; old/new-source A tables and payloads identical')


if __name__ == '__main__':
    if len(sys.argv) == 2:
        verify_packages(Path(sys.argv.pop()))
    unittest.main()
