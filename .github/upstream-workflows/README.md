# Parent-fork workflows

These files preserve the parent's automation as a reference. They are outside `.github/workflows/` and do not run in this fork.

The inherited builds package the NVIDIA frontend through a different toolchain and release process. They do not install this project's HIP toolchain or stage its model converter. Some also publish nightly releases or request the parent's signing service. RDNA2NR releases are assembled locally with `package_release.ps1` and checked before upload.

The active source-check workflow validates public documentation and rejects private model/driver assets. It does not claim to compile or qualify GPU execution on a hosted runner.
