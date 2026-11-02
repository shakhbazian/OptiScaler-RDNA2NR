"""Known model identities and serialized layer-layout templates.

The template numbers describe storage layouts, not execution order. Model B
has its own learned weights and graph; no layers are removed from model A.
"""
from dataclasses import dataclass
from pathlib import Path
import json

SOURCE_A = 'E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E'
SOURCE_AB = '2BC19BC09A7B89DD44F73E2AC0108FCC586E2154BDD0830F242F4F040E28789F'

@dataclass(frozen=True)
class ModelProfile:
    name: str
    graph: int
    resource: str
    resource_sha: str
    templates: tuple
    blend_hex: str

    @property
    def blocks(self):
        return len(self.templates)

PROFILES = {
    'original': ModelProfile('original', 1, 'WEIGHTS_HT',
        '836F445D06ECD2E59BB9F17B84B91C143396FD76CCDA1C9DC7FE81D5EDD548F4',
        tuple(range(71)), 'eb39'),
    'optimized': ModelProfile('optimized', 2, 'WEIGHTS_CC_SMALL_FP8',
        'F743E8EB320F42BC9BDC992EE2BF51F7BB7C8688CED4ECF5CE6807EC4EEF7962',
        (0, 1, 4, 5, 8, 9, 14, 15, 22, 23, 30, *range(31, 39),
         39, 40, 47, 48, 55, 56, 61, 62, 65, 66, 69, 70), '233a'),
}

# The installer and converter share these identities. C++ uses a generated copy.
CATALOG = json.loads(Path(__file__).with_name('model_catalog.json').read_text(encoding='utf-8'))
PACKAGES = {(r['sourceHash'], r['model']): (r['bytes'], r['packageHash']) for r in CATALOG}


def catalog_header():
    lines = ['#pragma once', '// Generated from tools/model_converter/model_catalog.json by model_profiles.py.',
             '#include <cstdint>', '#include <cstring>', 'namespace NrModel {',
             'struct PackageIdentity {const char* sourceHash; const char* packageHash; const char* relative; const char* legacyRelative; unsigned graph; std::uint64_t bytes;};',
             'inline constexpr PackageIdentity Packages[] = {']
    for r in CATALOG:
        lines.append('    {"%s","%s","%s","%s",%su,%sull},' %
                     (r['sourceHash'],r['packageHash'],r['relative'],r.get('legacyRelative',''),r['graph'],r['bytes']))
    lines += ['};', '// Header identity only; WeightPackage verifies the complete package SHA-256.',
              'inline const PackageIdentity* Identify(const void* header, std::uint64_t bytes) {',
              '    if(!header||bytes<128)return nullptr; const auto* p=static_cast<const unsigned char*>(header);',
              '    if(std::memcmp(p,"NRWGT001",8))return nullptr; std::uint32_t graph=0;std::memcpy(&graph,p+8,4);',
              '    static constexpr char hex[]="0123456789ABCDEF";char source[65]{};',
              '    for(unsigned i=0;i<32;++i){source[i*2]=hex[p[32+i]>>4];source[i*2+1]=hex[p[32+i]&15];}',
              '    for(const auto& item:Packages)if(item.graph==graph&&item.bytes==bytes&&!std::strcmp(item.sourceHash,source))return &item;',
              '    return nullptr;', '}', '} // namespace NrModel', '']
    return '\n'.join(lines)


def select_profile(source_hash, model):
    if source_hash not in (SOURCE_A, SOURCE_AB):
        raise ValueError('Unsupported source DLL SHA-256')
    profile = PROFILES[model]
    if profile.graph == 2 and source_hash != SOURCE_AB:
        raise ValueError('Optimized model B requires original DLL 310.9.16.0')
    return profile


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-catalog-header', type=Path, required=True)
    parser.parse_args().write_catalog_header.write_text(catalog_header(), encoding='utf-8')
