import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { EXRLoader } from 'three/examples/jsm/loaders/EXRLoader.js';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js';

// Raw PNG decoding logic utilizing browser DecompressionStream for exact pixel data
const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const RGBA_BYTES_PER_PIXEL = 4;

async function inflate(data) {
  const stream = new Response(data).body.pipeThrough(
    new DecompressionStream('deflate')
  );
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function concatChunks(chunks) {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const result = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}

function paethPredictor(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

function unfilterScanline(filterType, row, prevRow, out) {
  for (let i = 0; i < row.length; i++) {
    const a = i >= RGBA_BYTES_PER_PIXEL ? out[i - RGBA_BYTES_PER_PIXEL] : 0;
    const b = prevRow[i];
    const c = i >= RGBA_BYTES_PER_PIXEL ? prevRow[i - RGBA_BYTES_PER_PIXEL] : 0;
    const x = row[i];

    let value;
    switch (filterType) {
      case 0: // None
        value = x;
        break;
      case 1: // Sub
        value = x + a;
        break;
      case 2: // Up
        value = x + b;
        break;
      case 3: // Average
        value = x + Math.floor((a + b) / 2);
        break;
      case 4: // Paeth
        value = x + paethPredictor(a, b, c);
        break;
      default:
        throw new Error(`decodePngRgba8: unsupported filter type ${filterType}`);
    }
    out[i] = value & 0xff;
  }
}

async function decodePngRgba8(buffer) {
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);

  for (let i = 0; i < PNG_SIGNATURE.length; i++) {
    if (bytes[i] !== PNG_SIGNATURE[i]) {
      throw new Error('decodePngRgba8: not a valid PNG file');
    }
  }

  let width = 0;
  let height = 0;
  let bitDepth = 0;
  let colorType = 0;
  const idatChunks = [];

  let offset = PNG_SIGNATURE.length;
  while (offset < bytes.length) {
    const length = view.getUint32(offset, false);
    const type = String.fromCharCode(...bytes.subarray(offset + 4, offset + 8));
    const dataStart = offset + 8;

    if (type === 'IHDR') {
      width = view.getUint32(dataStart, false);
      height = view.getUint32(dataStart + 4, false);
      bitDepth = bytes[dataStart + 8];
      colorType = bytes[dataStart + 9];
    } else if (type === 'IDAT') {
      idatChunks.push(bytes.subarray(dataStart, dataStart + length));
    } else if (type === 'IEND') {
      break;
    }
    offset = dataStart + length + 4;
  }

  if (bitDepth !== 8 || colorType !== 6) {
    throw new Error(`decodePngRgba8: unsupported format (bitDepth=${bitDepth}, colorType=${colorType})`);
  }

  const raw = await inflate(concatChunks(idatChunks));
  const stride = width * RGBA_BYTES_PER_PIXEL;
  const data = new Uint8Array(width * height * RGBA_BYTES_PER_PIXEL);

  let prevRow = new Uint8Array(stride);
  let rawOffset = 0;
  for (let y = 0; y < height; y++) {
    const filterType = raw[rawOffset];
    rawOffset += 1;
    const row = raw.subarray(rawOffset, rawOffset + stride);
    rawOffset += stride;

    const outRow = data.subarray(y * stride, (y + 1) * stride);
    unfilterScanline(filterType, row, prevRow, outRow);
    prevRow = outRow;
  }

  return { data, width, height };
}

async function loadRawPngTexture(filePath) {
  const response = await fetch(filePath);
  if (!response.ok) {
    throw new Error(`Failed to fetch PNG from ${filePath}: ${response.statusText}`);
  }
  const { data, width, height } = await decodePngRgba8(await response.arrayBuffer());
  const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat, THREE.UnsignedByteType);
  texture.colorSpace = THREE.NoColorSpace;
  texture.flipY = false;
  texture.generateMipmaps = false;
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.NearestFilter;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.needsUpdate = true;
  return texture;
}

// Default 1x1 textures to bind when a texture isn't present
let defaultTexture = null;
function getDefaultTexture() {
  if (!defaultTexture) {
    const data = new Uint8Array([204, 204, 204, 255]); // grey
    defaultTexture = new THREE.DataTexture(data, 1, 1, THREE.RGBAFormat);

    defaultTexture.minFilter =
      defaultTexture.minFilter = THREE.NearestFilter;
    defaultTexture.generateMipmaps = false;

    defaultTexture.needsUpdate = true;
  }
  return defaultTexture;
}

export class VATLoader {
  static async load(rootPath, assetName, loadOptions = {}) {
    if (!rootPath.endsWith('/')) {
      rootPath += '/';
    }

    const isOpenVat = Boolean(loadOptions.isOpenVat);
    let metadataUrl;

    if (isOpenVat) {
      const baseName = loadOptions.gltfName || assetName.replace('_vat', '');
      metadataUrl = `${rootPath}${assetName}/${baseName}-remap_info.json`;
    } else {
      metadataUrl = `${rootPath}${assetName}/${assetName}_data.json`;
    }

    const metadataResponse = await fetch(metadataUrl);
    if (!metadataResponse.ok) {
      throw new Error(`VATLoader: failed to load metadata at: ${metadataUrl}`);
    }

    const metadataArray = await metadataResponse.json();
    const metadataRaw = Array.isArray(metadataArray) ? metadataArray[0] : metadataArray;
    const metadata = Object.assign({}, metadataRaw);

    let type = 'Softbody';
    let name = assetName;
    let isHdrVal = true;
    let framesVal = 1;
    let boundMinVal = new THREE.Vector3(0, 0, 0);
    let boundMaxVal = new THREE.Vector3(1, 1, 1);
    let axisSystemVal = 'Right-Handed Y-Up';

    if (isOpenVat) {
      // Decode OpenVAT properties
      type = 'Softbody'; // Blender OpenVAT currently exports morph softbody
      const remap = metadata['os-remap'] || {};
      framesVal = remap['Frames'] || 1;
      const minArr = remap['Min'] || [0, 0, 0];
      const maxArr = remap['Max'] || [1, 1, 1];
      boundMinVal.set(minArr[0], minArr[1], minArr[2]);
      boundMaxVal.set(maxArr[0], maxArr[1], maxArr[2]);
      isHdrVal = true; // OpenVAT always exports position as HDR EXR
      name = assetName;
    } else {
      const typeStr = metadata['VAT Type'] || 'Softbody';
      type = typeStr.charAt(0).toUpperCase() + typeStr.slice(1).toLowerCase();
      name = metadata['Name'] || assetName;
      framesVal = metadata['Frame Count'] || 1;
      boundMinVal.set(
        metadata['Bound Min X'] !== undefined ? metadata['Bound Min X'] : 0,
        metadata['Bound Min Y'] !== undefined ? metadata['Bound Min Y'] : 0,
        metadata['Bound Min Z'] !== undefined ? metadata['Bound Min Z'] : 0
      );
      boundMaxVal.set(
        metadata['Bound Max X'] !== undefined ? metadata['Bound Max X'] : 1,
        metadata['Bound Max Y'] !== undefined ? metadata['Bound Max Y'] : 1,
        metadata['Bound Max Z'] !== undefined ? metadata['Bound Max Z'] : 1
      );
      isHdrVal = Boolean(metadata['Use HDR Textures']);
      axisSystemVal = metadata['Axis System'] || 'Right-Handed Y-Up';
    }

    // Standard defaults
    const defaults = {
      additionalObjectSpaceOffset: new THREE.Vector3(0, 0, 0),
      additionalParticleScaleUniformMultiplier: 1.0,
      animateFirstFrame: false,
      boundMin: boundMinVal,
      boundMax: boundMaxVal,
      computeSpinfromHeadingVector: false,
      displayFrame: 0,
      enablePlayback: true,
      frameCount: framesVal,
      frameRate: metadata['Houdini FPS'] || 30,
      gameTimeAtFirstFrame: 0,
      globalParticlePiecesScaleMultiplier: 1,
      hideParticlesOverlappingObjectOrigin: true,
      inputTime: 0,
      instance: false,
      instanceCount: 0,
      instanceUpdateDynamicData: false,
      interframeInterpolation: true,
      interpolateColor: true,
      interpolateSpareColor: true,
      isColorTexHdr: true,
      isLookupTexHdr: false,
      isTexHdr: isHdrVal,
      noLerping: false,
      originEffectiveRadius: 1,
      particleHeightBaseScale: 0.5,
      particlePiecesScaleAreInPositionAlpha: false,
      particleShardCount: 0,
      particleShardIndex: 0,
      particleShards: false,
      particleSpinPhase: 0,
      particleTextureUScale: 1,
      particleTextureVScale: 1,
      particleWidthBaseScale: 0.5,
      perParticleRandomSpinSpeed: 0,
      perParticleRandomVelocityScale: 0,
      playbackSpeed: 1.0,
      scalebyVelocityAmount: 0,
      spinFromHeading: false,
      stretchByVelocity: false,
      stretchByVelocityAmount: 1.0,
      supportSurfaceNormalMaps: true,
      surfaceNormals: true,
      surfaceUVsfromColorRG: false,
      useAlphaForVelocityScale: false,
      useColorForVelocity: false,
      useCompressedNormals: true,
      useLookup: false,
      useParticleBillboarding: true,
      useParticleVelocitySpin: false,
      usePos2: false,
      axisSystem: axisSystemVal,
      useRightHandedCoordinates: axisSystemVal.startsWith('Right-Handed'),
      useSpareColor: false,
      vertexCount: metadata['Vertex Count'] || 0
    };

    // Variant overrides
    if (type === 'Dynamicmesh' || type === 'Fluid') {
      defaults.useLookup = true;
      defaults.noLerping = true;
      defaults.interframeInterpolation = false;
      defaults.supportSurfaceNormalMaps = true;
      defaults.surfaceNormals = true;
      defaults.useCompressedNormals = false;
    }

    // Merge metadata
    const overrides = Object.assign({}, defaults, {
      pivotMin: new THREE.Vector3(metadata['Pivot Min X'] !== undefined ? metadata['Pivot Min X'] : (metadata['Pivot Min'] !== undefined ? metadata['Pivot Min'] : 0), metadata['Pivot Min Y'] !== undefined ? metadata['Pivot Min Y'] : (metadata['Pivot Min'] !== undefined ? metadata['Pivot Min'] : 0), metadata['Pivot Min Z'] !== undefined ? metadata['Pivot Min Z'] : (metadata['Pivot Min'] !== undefined ? metadata['Pivot Min'] : 0)),
      pivotMax: new THREE.Vector3(metadata['Pivot Max X'] !== undefined ? metadata['Pivot Max X'] : (metadata['Pivot Max'] !== undefined ? metadata['Pivot Max'] : 1), metadata['Pivot Max Y'] !== undefined ? metadata['Pivot Max Y'] : (metadata['Pivot Max'] !== undefined ? metadata['Pivot Max'] : 1), metadata['Pivot Max Z'] !== undefined ? metadata['Pivot Max Z'] : (metadata['Pivot Max'] !== undefined ? metadata['Pivot Max'] : 1)),
      activePixelsRatio: (
        metadata['Active Pixels Ratio X'] !== undefined &&
        metadata['Active Pixels Ratio Y'] !== undefined
      ) ? new THREE.Vector2(
        metadata['Active Pixels Ratio X'],
        metadata['Active Pixels Ratio Y']
      ) : null,
      invertFrameV: loadOptions.invertFrameV !== undefined
        ? Boolean(loadOptions.invertFrameV)
        : Boolean(metadata['Invert Frame V']),
      particleShardCount: metadata['Particle Shard Count'] || 0,
      particlePiecesScaleAreInPositionAlpha: Boolean(metadata['Particle Pieces Scale Are In Position Alpha']),
      useCompressedNormals: metadata['Use Compressed Normals'] !== undefined
        ? Boolean(metadata['Use Compressed Normals'])
        : defaults.useCompressedNormals,
      useSpareColor: Boolean(metadata['Spare Color Texture']),
      usePos2: Boolean(metadata['Two Position Textures']),
      useLookup: metadata['Use Lookup Texture'] !== undefined ? Boolean(metadata['Use Lookup Texture']) : defaults.useLookup,
      dynamicMeshUsesPositionAlphaMask: Boolean(metadata['Dynamic Mesh Packed Position Alpha Mask']),
      dynamicMeshFrameVertexCounts: Array.isArray(metadata['Frame Vertex Counts']) ? metadata['Frame Vertex Counts'] : [],
      legacy: Boolean(metadata['Legacy Format'])
    });

    // 2. Load Mesh
    let mesh = null;
    let meshContainer = null;
    const meshAssetName = loadOptions.meshAssetName || assetName;

    if (meshAssetName.startsWith('demo_')) {
      const fbxLoader = new FBXLoader();
      const fbxUrl = `${rootPath}${meshAssetName}/${meshAssetName.replace('demo_', '')}_mesh.fbx`;
      const fbx = await fbxLoader.loadAsync(fbxUrl);
      meshContainer = fbx;
      fbx.traverse((node) => {
        if (!mesh && node.isMesh) {
          mesh = node;
        }
      });
      if (!mesh) {
        throw new Error(`VATLoader: no mesh found in FBX: ${fbxUrl}`);
      }
    } else {
      const gltfLoader = new GLTFLoader();
      const baseName = isOpenVat ? meshAssetName.replace('_vat', '') : meshAssetName;
      const gltfUrl = isOpenVat
        ? `${rootPath}${meshAssetName}/${baseName}.glb`
        : `${rootPath}${meshAssetName}/${meshAssetName}_mesh.glb`;
      console.log('VATLoader: loading GLTF from', gltfUrl, 'isOpenVat:', isOpenVat);
      const gltf = await gltfLoader.loadAsync(gltfUrl);
      meshContainer = gltf.scene;
      gltf.scene.traverse((node) => {
        if (!mesh && node.isMesh) {
          mesh = node;
        }
      });
      if (!mesh) {
        throw new Error(`VATLoader: no mesh found in GLB: ${gltfUrl}`);
      }
    }
    mesh.visible = false;

    if (mesh.geometry.attributes.uv1) {
      mesh.geometry.setAttribute('vatUv1', mesh.geometry.attributes.uv1);
      mesh.geometry.deleteAttribute('uv1');
    }
    if (mesh.geometry.attributes.uv2 && mesh.geometry.attributes.uv3) {
      const uv2 = mesh.geometry.attributes.uv2;
      const uv3 = mesh.geometry.attributes.uv3;
      const count = uv2.count;
      const pivotArr = new Float32Array(count * 3);
      for (let i = 0; i < count; i++) {
        // Original shader pivot encoding: vec3(uv2.x, uv3.x, uv3.y)
        // Variant 2 (VAT3 Rigidbody): position - vatPivot directly
        // Variant 5 (Legacy Rigidbody): position - vec3(vatPivot.x, vatPivot.y, 1.0 - vatPivot.z)
        pivotArr[i * 3 + 0] = uv2.getX(i);  // vatPivot.x = uv2.x
        pivotArr[i * 3 + 1] = uv3.getX(i);  // vatPivot.y = uv3.x
        pivotArr[i * 3 + 2] = uv3.getY(i);  // vatPivot.z = uv3.y
      }
      mesh.geometry.setAttribute('vatPivot', new THREE.BufferAttribute(pivotArr, 3));
      mesh.geometry.deleteAttribute('uv2');
      mesh.geometry.deleteAttribute('uv3');
    } else {
      if (mesh.geometry.attributes.uv2) {
        mesh.geometry.setAttribute('vatUv2', mesh.geometry.attributes.uv2);
        mesh.geometry.deleteAttribute('uv2');
      }
      if (mesh.geometry.attributes.uv3) {
        mesh.geometry.setAttribute('vatUv3', mesh.geometry.attributes.uv3);
        mesh.geometry.deleteAttribute('uv3');
      }
    }

    // 3. Load Textures
    const textureKeys = ['vatColTex', 'vatLookupTex', 'vatPos2Tex', 'vatPosTex', 'vatRotTex', 'vatSpareColTex'];
    const suffixMapping = {
      vatColTex: '_col',
      vatLookupTex: '_lookup',
      vatPos2Tex: '_pos2',
      vatPosTex: '_pos',
      vatRotTex: '_rot',
      vatSpareColTex: '_col2'
    };

    const textures = {};
    const loader = new THREE.TextureLoader();
    const exrLoader = new EXRLoader();

    for (const key of textureKeys) {
      const isRequired =
        key === 'vatPosTex' ||
        (key === 'vatColTex') ||
        (key === 'vatRotTex' && type !== 'Particles') ||
        (key === 'vatLookupTex' && overrides.useLookup && type === 'Dynamicmesh') ||
        (key === 'vatPos2Tex' && overrides.usePos2) ||
        (key === 'vatSpareColTex' && overrides.useSpareColor);

      if (isRequired) {
        const isColor = key === 'vatColTex' || key === 'vatSpareColTex';
        const isLookup = key === 'vatLookupTex';
        const useHdr = overrides.isTexHdr && !isLookup;
        const format = useHdr ? 'exr' : 'png';
        const suffix = suffixMapping[key];
        const baseName = assetName.startsWith('demo_') ? assetName.replace('demo_', '') : assetName;

        // OpenVAT exports a single position EXR named exactly after the asset (e.g. Cube_vat.exr)
        const path = (isOpenVat && key === 'vatPosTex')
          ? `${rootPath}${assetName}/${assetName}.${format}`
          : `${rootPath}${assetName}/${baseName}${suffix}.${format}`;

        try {
          let texture;
          if (key === 'vatLookupTex' && format === 'png') {
            texture = await loadRawPngTexture(path);
          } else {
            // Check if file exists and is not the SPA HTML fallback page
            const checkRes = await fetch(path, { method: 'HEAD' }).catch(() => null);
            if (!checkRes || !checkRes.ok || checkRes.headers.get('content-type')?.includes('text/html')) {
              throw new Error(`Texture not found (404/HTML fallback)`);
            }

            texture = format === 'exr'
              ? await exrLoader.loadAsync(path)
              : await loader.loadAsync(path);

            if (isColor) {
              texture.colorSpace = format === 'exr' ? THREE.LinearSRGBColorSpace : THREE.SRGBColorSpace;
            } else {
              texture.colorSpace = THREE.NoColorSpace;
            }
            texture.flipY = false;
            texture.generateMipmaps = false;
            texture.magFilter = THREE.NearestFilter;
            texture.minFilter = THREE.NearestFilter;
            texture.needsUpdate = true;
          }
          textures[key] = texture;
        } catch (err) {
          console.warn(`VATLoader: failed to load texture ${path}, using dummy.`, err);
          textures[key] = getDefaultTexture();
        }
      } else {
        textures[key] = getDefaultTexture();
      }
    }

    return {
      name,
      type,
      mesh,
      scene: meshContainer,
      textures,
      vatConfig: {
        staticInputs: overrides
      }
    };
  }
}
