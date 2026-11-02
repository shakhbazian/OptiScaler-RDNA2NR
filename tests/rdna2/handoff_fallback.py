"""A rejected pre-SR NR submission must publish this frame's clean SR input.

Requires a frontend built with -NativeTestHooks and source-built DX12 hosts.
Stage 1 recovers on the next frame. Stage 3 deliberately quarantines the owner;
that process is expected to report its terminal fault rather than a clean drain.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frontend', 'companion', 'model', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--reject-stage', choices=(1, 3), type=int, default=1)
    parser.add_argument('--expect-broken', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stage = out / 'stage'
    stage.mkdir()
    shutil.copy2(args.frontend, stage / 'OptiScaler.dll')
    shutil.copy2(args.companion, stage / 'dlssnr_hip_scheduled_bridge.dll')
    model = stage / 'model.nrwgt'
    try:
        os.link(args.model, model)
    except OSError:
        shutil.copy2(args.model, model)
    y, x = np.mgrid[:90, :160]
    pixels = np.stack((.15 + .6*x/160, .1 + .7*y/90,
                       .2 + .3*((x//8+y//8) % 2), np.ones_like(x)), axis=-1).astype('<f2')
    source = out / 'source.rgba.f16'
    pixels.tofile(source)
    cases = []
    for mode in (0, 1):
        target = out / str(mode)
        target.mkdir()
        (stage / 'OptiScaler.ini').write_text(
            '[Upscalers]\nDx12Upscaler=fsr22_12\n[DLSS]\nEnabled=false\n[DlssNr]\n'
            f'Enabled={str(bool(mode)).lower()}\nBackend=4\nHipModel=b\nModelBPath={model}\n'
            'RunBeforeSR=true\nWorkingScale=0.75\nStyle=0\nLocalTone=1\nLocalStructure=1\n'
            'SkinStructure=-1\nAutoMask=true\nIntensity=1\nTemporalAccumulation=true\nApplyModel=true\n'
            '[Hotfix]\nColorResourceBarrier=64\nOutputResourceBarrier=8\nMotionVectorResourceBarrier=64\n'
            '[ProcessFilter]\nProcessExclusionList=optiscaler_dx12_evaluate_host.exe\n', encoding='utf-8-sig')
        env = os.environ.copy()
        env.update(PATH=os.environ['WINDIR'] + '/System32;' + os.environ['WINDIR'],
                   NR_HOST_OUTPUT_DIR=str(target), NR_HOST_REJECT_FRAME='1',
                   NR_HOST_REJECT_STAGE=str(args.reject_stage), NR_HOST_REFERENCE_SCALE='0.75')
        result = subprocess.run([str(repo / 'build/tests/rdna2/optiscaler_dx12_evaluate_host.exe'),
                                 str(stage / 'OptiScaler.dll'), str(source), '160', '90', str(mode),
                                 'pan_jitter', '4'], cwd=repo, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        (target / 'run.log').write_bytes(result.stdout)
        expected_fault = mode == 1 and args.reject_stage == 3 and result.returncode != 0 and (
            b'NativeExpectedFault verified=1' in result.stdout and
            b'FAIL OptiScaler native DX12 Evaluate: expected abort quarantine' in result.stdout)
        if result.returncode and not expected_fault:
            raise RuntimeError(f'mode {mode} failed; see {target / "run.log"}')
        # That terminal coverage check follows every per-frame debug validation.
        assert expected_fault or b'debug_clean=1' in result.stdout
        cases.append({'mode': mode, 'expectedQuarantine': bool(expected_fault), 'exitCode': result.returncode})
    clean = (out / '0/1.rgba.f16').read_bytes()
    rejected = (out / '1/1.rgba.f16').read_bytes()
    record = {'rejectStage': args.reject_stage, 'cases': cases, 'rawFramePreserved': clean == rejected,
              'offSha256': hashlib.sha256(clean).hexdigest(),
              'rejectedSha256': hashlib.sha256(rejected).hexdigest()}
    (out / 'result.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))
    assert (clean != rejected) if args.expect_broken else (clean == rejected)


if __name__ == '__main__':
    main()
