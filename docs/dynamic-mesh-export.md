# Dynamic Mesh Export Notes

This document describes the current Blender-to-`three-vat` dynamic mesh export contract used by the in-progress exporter in [vat_export.py](/C:/work/vat3-webgl/vat_export.py).

It is intentionally practical rather than formal. The goal is to capture the assumptions that currently make the exported fluid/dynamic-mesh samples play correctly in the runtime.

## Overview

For dynamic mesh / fluid playback, the runtime treats the exported mesh as a stable carrier for addressing VAT data, not as the animated shape itself.

That means:

- the GLB mesh is a triangle soup with a fixed maximum triangle budget
- animated positions come from the VAT textures
- animated normals come from the VAT rotation/normal texture
- packed exports no longer depend on a giant `_lookup` texture
- the loader/runtime uses metadata to decide whether the asset uses the legacy lookup path or the newer packed atlas path

## Files

A packed Blender-authored dynamic mesh export currently looks like:

- `<assetName>_data.json`
- `<assetName>_mesh.glb`
- `<assetName>_pos.exr` or `<assetName>_pos.png`
- `<assetName>_rot.exr` or `<assetName>_rot.png`

Legacy-style dynamic mesh samples may also include:

- `<assetName>_lookup.png`
- `<assetName>_col.exr` or `.png`

The newer packed exporter path does not require `_lookup`.

## Carrier Mesh

The exported GLB mesh is a generated triangle soup sized to the maximum triangle count observed across the animation range.

Important properties:

- vertices are unwelded
- triangles are emitted as fixed triplets
- the mesh is a stable indexing surface, not a sampled frame of the final sim
- `uv.x` encodes the local column inside the packed texture atlas
- `uv.y` encodes the local row within a frame block

The runtime adds the frame offset itself when sampling the packed atlas.

## Packed Atlas Layout

For packed dynamic mesh exports:

- texture width is bounded by the exporter `Max Texture Width`
- texture width is forced to be divisible by `3` so triangle triplets do not straddle row boundaries awkwardly
- each frame occupies `rowsPerFrame = ceil(maxVertexCount / textureWidth)` rows
- the full texture height is `frameCount * rowsPerFrame`

The exporter writes vertex slots linearly into that atlas:

- `vertexIndex -> row = frameRowStart + floor(vertexIndex / textureWidth)`
- `vertexIndex -> col = vertexIndex % textureWidth`

## Position Alpha Mask

Packed dynamic mesh exports now use the position texture alpha channel as an explicit validity mask.

- `pos.rgb` stores the packed position payload
- `pos.a = 1.0` for a real exported vertex slot
- `pos.a = 0.0` for padded / unused atlas texels

This is more robust than inferring validity only from a frame-wide vertex count, because the shader can decide directly from the sampled texels whether a triangle slot is real.

## Metadata

The packed exporter currently writes a few dynamic-mesh-specific metadata fields:

- `"Use HDR Textures"`: whether `pos` / `rot` are written as EXR or quantized PNG
- `"Use Lookup Texture"`: `false` for the newer packed atlas path
- `"Dynamic Mesh Packed Position Alpha Mask"`: `true` when the packed position alpha mask is present
- `"Frame Vertex Counts"`: per-frame active vertex counts, still useful for debugging and fallback logic
- `"Axis System"`: exporter-declared source axis system, such as `Right-Handed Y-Up` or `Right-Handed Z-Up`

## Runtime Expectations

The current runtime expects the following for packed dynamic mesh playback:

- the loader reads metadata and selects packed vs lookup behavior
- the mesh UVs refer to local slot coordinates, not full-atlas frame coordinates
- the shader computes the current frame row offset
- the shader samples `pos` / `rot` directly from the packed atlas
- the shader uses the position alpha mask to decide whether a slot is active

## Known Limits

This path is working, but still evolving.

Current caveats:

- dynamic remesh correspondence is still heuristic
- triangle ordering is stabilized with centroid sorting, but this is not guaranteed to match Houdini exactly
- large dense frames stress both export time and texture memory
- the legacy `demo_fluid` sample still uses a different lookup-driven path

## Recommended Next Steps

- document the exact JSON schema more formally once it stabilizes
- compare the packed Blender path against Houdini exports more directly
- audit whether the legacy `demo_fluid` sample would benefit from the same cleanup ideas
- validate the same export/runtime conventions on additional fluid sims with different remesh behavior
