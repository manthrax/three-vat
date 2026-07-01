## three-vat

`three-vat` is a Three.js library for importing and playing Houdini-style Vertex Animation Textures in the browser.

The goal is to provide a robust, practical runtime for VAT-driven effects in Three.js:

- loading sidecar metadata plus mesh/texture assets from disk
- patching standard Three.js materials with VAT deformation via `onBeforeCompile`
- supporting instancing with per-instance time offsets and speed multipliers
- covering multiple VAT variants: soft body, dynamic mesh/fluid, rigid body, particles, and an OpenVAT sample

Demo: [https://manthrax.github.io/three-vat/](https://manthrax.github.io/three-vat/)

![Demo screenshot](https://github.com/user-attachments/assets/5e51d936-4f44-4ee2-a80c-1f9c3858f62b)

## Philosophy

The core idea is to make VAT playback feel like a normal part of a Three.js pipeline, not a sidecar experiment that requires replacing everything with custom shaders.

In practice, that means:

- preserving standard Three.js material workflows as much as possible
- supporting multiple VAT flavors behind one runtime model
- aiming for compatibility with Houdini-style VAT3 asset conventions
- leaving room for Blender-authored exports that target the same runtime contract

This repo includes a demo app, runtime code, and an in-progress Blender exporter because the longer-term goal is a usable end-to-end pipeline, not just an isolated viewer.

## Current shape of the repo

- [index.html](/C:/work/vat3-webgl/index.html) is the standalone demo app and scene setup
- [VATLoader.js](/C:/work/vat3-webgl/VATLoader.js) loads metadata, mesh assets, and VAT textures
- [VATEffect.js](/C:/work/vat3-webgl/VATEffect.js) injects shader logic into Three.js materials
- [vat_export.py](/C:/work/vat3-webgl/vat_export.py) is a Blender addon/exporter for generating VAT assets
- [public/examples](/C:/work/vat3-webgl/public/examples) contains sample datasets for each supported variant

## Quick start

```bash
npm install
npm run dev
```

Build for the GitHub Pages-style deployment target:

```bash
npm run build
```

Preview the production build locally:

```bash
npm run preview
```

## Runtime usage

The demo uses the loader/effect pair like this:

```js
import { VATLoader } from './VATLoader.js';
import { VATEffect } from './VATEffect.js';

const assets = await VATLoader.load('examples/', 'vat3_softBody');
const vat = new VATEffect(assets, { camera });

scene.add(vat.mesh);

vat.speed = 1.0;
vat.time = 0.0;
vat.setEnabled(true);
```

For instanced playback, the runtime supports per-instance offsets and rate variation:

```js
vat.setInstanceTimeOffset(index, offsetInSeconds);
vat.setInstanceSpeedScale(index, speedMultiplier);
```

## Supported asset conventions

The loader expects a folder per asset under `public/examples/<assetName>/`.

### Standard Houdini-style export

Expected files typically look like:

- `<assetName>_data.json`
- `<assetName>_mesh.glb` or legacy `*_mesh.fbx` for the older demo assets
- `<assetName>_pos.exr` or `.png`
- `<assetName>_rot.exr` or `.png` when the variant needs rotation/normal data
- optional `_col`, `_lookup`, `_pos2`, `_col2`

### OpenVAT sample

The repo also contains an OpenVAT-style sample (`Cube_vat`) with a slightly different naming/layout convention:

- metadata is read from `<baseName>-remap_info.json`
- the mesh is read from `<baseName>.glb`
- the primary position texture is read from `<assetName>.exr`

## VAT variants in this repo

### Dynamic mesh / fluid

- legacy samples may use a lookup texture indirection path
- current Blender-packed exports use a generated triangle-soup carrier mesh plus packed position/rotation atlases
- positions come from `_pos`
- rotations/normals come from `_rot`
- dynamic topology may use `_lookup` or the newer packed-atlas path depending on metadata

See [docs/dynamic-mesh-export.md](/C:/work/vat3-webgl/docs/dynamic-mesh-export.md) for the current packed dynamic-mesh export/runtime contract.

### Soft body

- standard indexed or unindexed deforming mesh
- texture width corresponds to vertex count
- texture height corresponds to frame count
- positions come from `_pos`
- orientation/normal data comes from `_rot`

### Particles

- texture width corresponds to maximum particle count
- positions come from `_pos`
- this sample also uses `_col` as the particle/albedo texture
- the runtime can billboard particles and vary playback per instance

### Rigid body

- chunk transforms are sampled per frame
- pivots are reconstructed from mesh UV channels when present
- positions come from `_pos`
- rotations come from `_rot`

## Blender exporter

[vat_export.py](/C:/work/vat3-webgl/vat_export.py) registers a Blender sidebar panel named `VAT Exporter` and can export:

- dynamic / fluid mesh
- soft body
- rigid body / chunk animation
- particles

The exporter writes Houdini-style sidecar JSON plus EXR/PNG textures and is a good foundation for documenting a repeatable authoring pipeline next.

For the current dynamic mesh / fluid export details, see [docs/dynamic-mesh-export.md](/C:/work/vat3-webgl/docs/dynamic-mesh-export.md).

## Status

The runtime is already useful as a reference implementation and demo, but the project is still evolving. Expect the API, exporter, and asset conventions to continue getting refined as the library moves toward a cleaner public surface.

## Roadmap

- tighten the loader/runtime API into a more library-shaped public surface
- document the asset contract more formally per VAT variant
- improve Blender exporter compatibility with Houdini-style VAT3 conventions
- expand sample coverage and validation across soft body, rigid body, particles, and dynamic mesh/fluid workflows
- reduce special-case handling between legacy samples, current samples, and OpenVAT-style inputs

## References

This implementation was informed by:

- [mikelyndon/r3f-webgl-vertex-animation-textures](https://github.com/mikelyndon/r3f-webgl-vertex-animation-textures)
- [floating-world-lda/vat3-wgsl-ts (Three.js)](https://github.com/floating-world-lda/vat3-wgsl-ts/tree/main/three-js)
- [floating-world-lda/vat3-wgsl-ts (Babylon.js)](https://github.com/floating-world-lda/vat3-wgsl-ts/tree/main/babylon-js)
- [Babylon playground reference](https://playground.babylonjs.com/#XKW2C5)

by Thrax



