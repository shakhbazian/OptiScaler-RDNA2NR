# Frame hold

D3D12 hold freezes owned colour, depth, motion, exposure and sampling metadata while model/composition settings change. Original game resources and parameter bindings are preserved. Release resumes live input.

Shape or placement changes invalidate the capture. Reset discards unsubmitted captures; GPU completion protects retirement. Finished-picture hold uses one clean snapshot per feature, independent of rotating presentation buffers, with fence-protected reuse.

The game simulation continues while the NR image is frozen. Native Vulkan does not implement this input-hold path. See [presentation bridges](../../../docs/upstream/NR-FINISHED-BRIDGES.md).
