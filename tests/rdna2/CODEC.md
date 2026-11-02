# Colour-codec regression test

`Build-Hosts.ps1` builds `codec_identity.exe`. It runs the committed GPU shader
without HIP or a model: encode the supplied frame, optionally resize its proxy,
use that proxy as the model's answer, and resolve onto the original frame.
The D3D12 debug layer must be installed. Every warning or error fails the test.

```powershell
tests/rdna2/Build-Hosts.ps1
build/tests/rdna2/codec_identity.exe input.rgba.f16 256 64 output.rgba.f16 0 .75 1 skin-half
```

The input is tightly packed, nonnegative RGBA16F. Arguments after the output are
the linear-HDR flag (`0` means display-referred), working scale, and whether to
clamp the fake model answer to `[0,1]`. The optional final argument selects a
check:

| Profile | Assertion |
| --- | --- |
| `default` | Identity preserves every RGBA bit. |
| `skin-half` | Partial skin/environment attenuation preserves identity. |
| `skin-zero` | Fully protected pixels preserve identity. |
| `replace-native` | Replace mode with inactive native-scale sharpening preserves identity. |
| `lighting` | Relative-lighting reconstruction preserves identity at reduced scale. |
| `skin-mask` | A nonzero edit still produces a bounded grey mask with original alpha. |
| `wipe` | Identity preserves both sides; the divider remains visible. |
| `ulp` | A one-FP16-ULP red-channel edit affects the result. Supply a varied frame with positive midtones. |

Include both bounded colours and values above one, almost-black colours such as
`(0,0,2^-20)`, and non-opaque alpha. Repeat at native and reduced scale, with both
HDR flags. These are shader checks; model quality and temporal behaviour need
separate tests with actual model output.
