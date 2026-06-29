import * as THREE from 'three';

// Common GLSL functions for Houdini VAT3
const VAT3_GLSL_COMMON = `
#define tau 6.28318530718

#ifndef VAT_ACTIVE_TIME
#define VAT_ACTIVE_TIME u_inputTime
#endif

// Uniforms
uniform sampler2D u_vatColTex;
uniform sampler2D u_vatLookupTex;
uniform sampler2D u_vatPos2Tex;
uniform sampler2D u_vatPosTex;
uniform sampler2D u_vatRotTex;
uniform sampler2D u_vatSpareColTex;

// Dynamic inputs
uniform float u_time;
uniform float u_playbackSpeed;
uniform bool u_enablePlayback;
uniform mat4 u_modelViewMatrix;
uniform mat4 u_viewToModelMatrix;

// Static inputs
uniform vec3 u_additionalObjectSpaceOffset;
uniform float u_additionalParticleScaleUniformMultiplier;
uniform bool u_animateFirstFrame;
uniform vec3 u_boundMax;
uniform vec3 u_boundMin;
uniform bool u_computeSpinfromHeadingVector;
uniform float u_displayFrame;
uniform float u_frameCount;
uniform float u_frameRate;
uniform float u_gameTimeAtFirstFrame;
uniform float u_globalParticlePiecesScaleMultiplier;
uniform bool u_hideParticlesOverlappingObjectOrigin;
uniform float u_inputTime;
uniform bool u_instance;
uniform float u_instanceCount;
uniform bool u_instanceUpdateDynamicData;
uniform bool u_interframeInterpolation;
uniform bool u_interpolateColor;
uniform bool u_interpolateSpareColor;
uniform bool u_isColorTexHdr;
uniform bool u_isLookupTexHdr;
uniform bool u_isTexHdr;
uniform bool u_noLerping;
uniform float u_originEffectiveRadius;
uniform float u_particleHeightBaseScale;
uniform bool u_particlePiecesScaleAreInPositionAlpha;
uniform float u_particleShardCount;
uniform float u_particleShardIndex;
uniform bool u_particleShards;
uniform float u_particleSpinPhase;
uniform float u_particleTextureUScale;
uniform float u_particleTextureVScale;
uniform float u_particleWidthBaseScale;
uniform float u_perParticleRandomSpinSpeed;
uniform float u_perParticleRandomVelocityScale;
uniform float u_scalebyVelocityAmount;
uniform bool u_spinFromHeading;
uniform bool u_stretchByVelocity;
uniform float u_stretchByVelocityAmount;
uniform bool u_supportSurfaceNormalMaps;
uniform bool u_surfaceNormals;
uniform bool u_surfaceUVsfromColorRG;
uniform bool u_useAlphaForVelocityScale;
uniform bool u_useColorForVelocity;
uniform bool u_useCompressedNormals;
uniform bool u_useLookup;
uniform bool u_useParticleBillboarding;
uniform bool u_useParticleVelocitySpin;
uniform bool u_usePos2;
uniform bool u_useRightHandedCoordinates;
uniform bool u_useSpareColor;
uniform float u_vertexCount;

// Structs
struct Vat3_AnimationData {
  float animProgressNextFrame;
  float animProgressThisFrame;
  float currentFrame;
  float currentFramePlusOne;
  float frameCount;
  float frameCountInverse;
  float loopedAnimFrame;
  float timeElapsed;
};

struct Vat3_UVs {
  vec2 nextFrameUv;
  vec2 thisFrameUv;
};

struct Vat3_TexturesData {
  vec4 colorThisFrame;
  vec4 colorNextFrame;
  vec4 posThisFrame;
  vec4 posNextFrame;
  vec4 rotThisFrame;
  vec4 rotNextFrame;
  vec4 lookupThisFrame;
  vec4 lookupNextFrame;
  vec3 pos2ThisFrame;
  vec3 pos2NextFrame;
  vec4 spareColorThisFrame;
  vec4 spareColorNextFrame;
};

struct Vat3_VelocityData {
  vec3 direction;
  vec3 velocity;
};

struct Vat3_ParticleDirection {
  vec3 particleRightDir;
  vec3 particleUpDir;
};

struct Vat3_PScaleData {
  float clampedPScaleNextFrame;
  float clampedPScaleThisFrame;
  float maxDataPScale;
  float posAlphaNextFrame;
  float posAlphaThisFrame;
  float pScaleNextFrame;
  float pScaleThisFrame;
};

struct Vat3_ParticleScale {
  float baseScale;
  float finalScale;
};

struct Vat3_SurfaceData {
  vec3 normal;
  vec3 tangent;
};

struct Vat3_Outputs {
  vec3 outAnimationProgress;
  vec4 outColorAndAlpha;
  vec3 outNormal;
  vec3 outPosition;
  vec4 outSpareColorAndAlpha;
  vec3 outTangent;
  vec2 surfaceUv;
};

// Functions
float scalarMod(float a, float b) {
  return a - b * floor(a / b);
}

vec2 computeActivePixelsRatio(vec3 bMin, vec3 bMax) {
  vec3 scaledMin = bMin * 10.0;
  vec3 scaledMax = bMax * 10.0;
  
  // Round to 4 decimal places to prevent float precision noise from triggering ceil()
  scaledMin = floor(scaledMin * 10000.0 + 0.5) / 10000.0;
  scaledMax = floor(scaledMax * 10000.0 + 0.5) / 10000.0;
  
  float ratioX = 1.0 - (ceil(scaledMin.z) - scaledMin.z);
  float ratioY = 1.0 - (ceil(scaledMax.x) - scaledMax.x);
  return vec2(ratioX, ratioY);
}

vec4 decodeQuaternion(vec3 XYZ, int maxComponent) {
  float w = sqrt(max(0.0, 1.0 - dot(XYZ, XYZ)));
  if (maxComponent == 0) return vec4(XYZ.x, XYZ.y, XYZ.z, w);
  if (maxComponent == 1) return vec4(w, XYZ.y, XYZ.z, XYZ.x);
  if (maxComponent == 2) return vec4(XYZ.x, w, XYZ.z, XYZ.y);
  if (maxComponent == 3) return vec4(XYZ.x, XYZ.y, w, XYZ.z);
  return vec4(XYZ.x, XYZ.y, XYZ.z, w);
}

void decodeRotationTexture(vec4 rotTexData, vec3 normalDefaults, vec3 tangentDefaults, out vec3 normal, out vec3 tangent) {
  vec3 crossNormal = cross(rotTexData.xyz, normalDefaults);
  vec3 normalLengMul = rotTexData.www * normalDefaults;
  vec3 normalToUnpack = cross(rotTexData.xyz, normalLengMul + crossNormal);
  normal = normalToUnpack * 2.0 + normalDefaults;

  vec3 crossTangent = cross(rotTexData.xyz, tangentDefaults);
  vec3 tangentLengMul = rotTexData.www * tangentDefaults;
  vec3 tangentToUnpack = cross(rotTexData.xyz, tangentLengMul + crossTangent);
  tangent = tangentToUnpack * 2.0 + tangentDefaults;
}

vec3 decodeRotationTextureNoTangent(vec4 rotTexData, vec3 normalDefaults) {
  vec3 crossNormal = cross(rotTexData.xyz, normalDefaults);
  vec3 normalLengMul = rotTexData.www * normalDefaults;
  vec3 normalToUnpack = cross(rotTexData.xyz, normalLengMul + crossNormal);
  return normalToUnpack * 2.0 + normalDefaults;
}

vec4 vatLoad(sampler2D tex, vec2 uv) {
  ivec2 sz = textureSize(tex, 0);
  return texelFetch(tex, clamp(ivec2(vec2(sz) * uv), ivec2(0), sz - 1), 0);
}

vec2 decodeVatLookupUv(vec4 lookupSample, bool isLookupHdr) {
  float maxRange = isLookupHdr ? 2048.0 : 255.0;
  float u = lookupSample.x + lookupSample.y / maxRange;
  float v = lookupSample.z + lookupSample.w / maxRange;
  return vec2(u, v);
}

vec3 decodePosition(vec4 posTex, vec3 inputBoundsRange, vec3 boundMin, bool isTexHdr) {
  vec3 pos = posTex.xyz;
  if (!isTexHdr) {
    pos = pos * inputBoundsRange + boundMin;
  }
  return pos;
}

//float randomFloat(vec2 seed) {
//  return fract(sin(dot(seed, vec2(22.9898, 178.24313))) * 12858.24161);
//}

vec3 recoverCompressedNormal(float normalInAlpha) {
  float highRange = normalInAlpha * 1024.0;
  float lowRange = floor(normalInAlpha * 32.0);
  vec2 angleToUnpack = vec2(lowRange / 31.5, (highRange - (lowRange * 32.0)) / 31.5);
  vec2 unpackedAngle = angleToUnpack * 4.0 - vec2(2.0, 2.0);
  float dotSquare = dot(unpackedAngle, unpackedAngle);
  float dotSquareRemaped = sqrt(1.0 - (dotSquare * 0.25));
  vec2 normalXZ = dotSquareRemaped * unpackedAngle;
  vec3 unclampedNormal = vec3(normalXZ.x, 1.0 - (dotSquare * 0.5), normalXZ.y);
  return clamp(unclampedNormal, vec3(-1.0, -1.0, -1.0), vec3(1.0, 1.0, 1.0));
}

vec3 rotateVectorByQuaternion(vec3 vect, vec4 quat) {
  vec3 crossXYZ = cross(quat.xyz, vect);
  vec3 quatWvec = quat.www * vect;
  return cross(quat.xyz, crossXYZ + quatWvec) * 2.0 + vect;
}

Vat3_AnimationData computeAnimationData() {
  float timeElapsed = VAT_ACTIVE_TIME - u_gameTimeAtFirstFrame;
  float animationProgress = (u_frameRate / (u_frameCount - 0.01)) * timeElapsed;
  float loopedAnimFrame = fract(animationProgress * u_playbackSpeed) * u_frameCount;
  
  float currentFrame = u_enablePlayback ? loopedAnimFrame : u_displayFrame;
  float currentFramePlusOne = u_enablePlayback ? floor(loopedAnimFrame + 1.0) : floor(u_displayFrame + 1.0);
  
  float frameCount = u_frameCount;
  float frameCountInverse = 1.0 / frameCount;
  float animProgressThisFrame = scalarMod(currentFramePlusOne - 1.0, frameCount) * frameCountInverse;
  float animProgressNextFrame = u_noLerping ? animProgressThisFrame : scalarMod(currentFramePlusOne, frameCount) * frameCountInverse;

  return Vat3_AnimationData(
    animProgressNextFrame,
    animProgressThisFrame,
    currentFrame,
    currentFramePlusOne,
    frameCount,
    frameCountInverse,
    loopedAnimFrame,
    timeElapsed
  );
}

Vat3_UVs computeUVs(vec2 texCoord, Vat3_AnimationData animData, bool isHdr, vec2 activePixelsRatio) {
  float scaledU = texCoord.x * activePixelsRatio.x;
  float samplingVThisFrame = (texCoord.y + animData.animProgressThisFrame) * activePixelsRatio.y;
  float samplingVNextFrame = (texCoord.y + animData.animProgressNextFrame) * activePixelsRatio.y;

  float clampedThisFrameV = fract(samplingVThisFrame);
  float clampedNextFrameV = fract(samplingVNextFrame);

  float finalThisFrameV = isHdr ? (1.0 - clampedThisFrameV) : clampedThisFrameV;
  float finalNextFrameV = isHdr ? (1.0 - clampedNextFrameV) : clampedNextFrameV;

  return Vat3_UVs(
    vec2(scaledU, finalNextFrameV),
    vec2(scaledU, finalThisFrameV)
  );
}

Vat3_TexturesData sampleVat3TexturesFromParams(
  vec2 thisFrameUv, vec2 nextFrameUv, bool noLerp,
  bool useLookup, bool usePos2, bool useSpareColor
) {
  Vat3_TexturesData result;

  result.colorThisFrame = vatLoad(u_vatColTex, thisFrameUv);
  result.posThisFrame   = vatLoad(u_vatPosTex, thisFrameUv);
  result.rotThisFrame   = vatLoad(u_vatRotTex, thisFrameUv);

  if (noLerp) {
    result.colorNextFrame = result.colorThisFrame;
    result.posNextFrame   = result.posThisFrame;
    result.rotNextFrame   = result.rotThisFrame;
  } else {
    result.colorNextFrame = vatLoad(u_vatColTex, nextFrameUv);
    result.posNextFrame   = vatLoad(u_vatPosTex, nextFrameUv);
    result.rotNextFrame   = vatLoad(u_vatRotTex, nextFrameUv);
  }

  if (useLookup) {
    result.lookupThisFrame = vatLoad(u_vatLookupTex, thisFrameUv);
    result.lookupNextFrame = noLerp ? result.lookupThisFrame : vatLoad(u_vatLookupTex, nextFrameUv);
  }

  if (usePos2) {
    result.pos2ThisFrame = vatLoad(u_vatPos2Tex, thisFrameUv).xyz;
    result.pos2NextFrame = noLerp ? result.pos2ThisFrame : vatLoad(u_vatPos2Tex, nextFrameUv).xyz;
  }

  if (useSpareColor) {
    result.spareColorThisFrame = vatLoad(u_vatSpareColTex, thisFrameUv);
    result.spareColorNextFrame = noLerp ? result.spareColorThisFrame : vatLoad(u_vatSpareColTex, nextFrameUv);
  }

  return result;
}

vec4 interpolateColor(
  vec4 colorThisFrame,
  vec4 colorNextFrame,
  float interpolationAlpha,
  bool interframeInterpolation,
  bool doInterpolateColor
) {
  return (interframeInterpolation && doInterpolateColor) ? mix(colorThisFrame, colorNextFrame, interpolationAlpha) : colorThisFrame;
}

vec3 interpolateVector3(
  vec3 vectorThisFrame,
  vec3 vectorNextFrame,
  float interpolationAlpha,
  bool interframeInterpolation,
  bool doInterpolateVector
) {
  return (interframeInterpolation && doInterpolateVector) ? mix(vectorThisFrame, vectorNextFrame, interpolationAlpha) : vectorThisFrame;
}

Vat3_SurfaceData computeTextureBasedNormals(
  Vat3_AnimationData animData,
  Vat3_TexturesData textures,
  float interpolationAlpha,
  vec3 finalVertexPosition
) {
  Vat3_SurfaceData surfaceData;
  
  if (u_useCompressedNormals) {
    surfaceData.normal = normalize(recoverCompressedNormal(textures.posThisFrame.a));
    surfaceData.tangent = vec3(0.0);
  } else {
    vec4 rotThisFrameToDecode = u_isTexHdr ? textures.rotThisFrame : (textures.rotThisFrame - 0.5) * 2.0;
    vec4 rotNextFrameToDecode = u_isTexHdr ? textures.rotNextFrame : (textures.rotNextFrame - 0.5) * 2.0;
    
    vec3 defaultNormal = vec3(0.0, 1.0, 0.0);
    vec3 defaultTangent = vec3(-1.0, 0.0, 0.0);

    vec3 normalThis, tangentThis;
    decodeRotationTexture(rotThisFrameToDecode, defaultNormal, defaultTangent, normalThis, tangentThis);

    vec3 normalNext, tangentNext;
    decodeRotationTexture(rotNextFrameToDecode, defaultNormal, defaultTangent, normalNext, tangentNext);
    
    if (u_interframeInterpolation) {
      surfaceData.normal = mix(normalThis, normalNext, interpolationAlpha);
      surfaceData.tangent = mix(tangentThis, tangentNext, interpolationAlpha);
    } else {
      surfaceData.normal = normalThis;
      surfaceData.tangent = tangentThis;
    }
    
    surfaceData.normal = normalize(surfaceData.normal);
    surfaceData.tangent = u_supportSurfaceNormalMaps ? normalize(surfaceData.tangent) : vec3(0.0);
  }

  return surfaceData;
}

vec2 resolveVatSurfaceUv(vec4 colorPayload, float posAlphaThisFrame, bool useColorRg) {
  vec2 uvFromColor = vec2(colorPayload.r, 1.0 - colorPayload.g);
  vec2 uvFromPosition = vec2(posAlphaThisFrame, 1.0 - colorPayload.a);
  return useColorRg ? uvFromColor : uvFromPosition;
}

bool isVatPaddingVertex(vec2 texCoord, float frameCount) {
  return texCoord.y > (1.0 - 0.5 / frameCount);
}

Vat3_VelocityData calculateVelocity(
  vec3 nextFramePos, vec3 thisFramePos,
  vec4 interpolatedColor, mat4 modelViewMatrix,
  float perParticleRandomVelocityScale,
  bool useAlphaForVelocityScale, bool useColorForVelocity,
  bool useInvertedColorR
) {
  vec3 velocity;
  if (useColorForVelocity) {
    float r = useInvertedColorR ? -interpolatedColor.r : interpolatedColor.r;
    vec3 baseVelocity = vec3(r, interpolatedColor.g, interpolatedColor.b);
    velocity = useAlphaForVelocityScale ? baseVelocity * interpolatedColor.a : baseVelocity;
  } else {
    vec3 posDelta = nextFramePos - thisFramePos;
    vec3 viewPosDelta = (modelViewMatrix * vec4(posDelta, 0.0)).xyz;
    velocity = vec3(viewPosDelta.xy, 0.0);
  }
  
  velocity = velocity * perParticleRandomVelocityScale;
  vec3 direction = length(velocity) > 0.0 ? normalize(velocity) : vec3(0.0);
  
  return Vat3_VelocityData(direction, velocity);
}

Vat3_ParticleDirection calculateParticleOrientation(Vat3_VelocityData velocityData) {
  vec3 particleRightDir;
  vec3 particleUpDir;
  
  if (u_useParticleBillboarding) {
    vec3 camRight = normalize((u_viewToModelMatrix * vec4(1.0, 0.0, 0.0, 0.0)).xyz);
    vec3 camUp = normalize((u_viewToModelMatrix * vec4(0.0, 1.0, 0.0, 0.0)).xyz);
    particleRightDir = -camRight;
    particleUpDir = camUp;
  } else {
    particleRightDir = vec3(-1.0, 0.0, 0.0);
    particleUpDir = vec3(0.0, 1.0, 0.0);
  }
  
  if (u_useParticleVelocitySpin) {
    float velocityMagnitude = length(velocityData.velocity);
    float spinAngle = velocityMagnitude * u_perParticleRandomSpinSpeed;
    float cosSpin = cos(spinAngle);
    float sinSpin = sin(spinAngle);
    
    vec3 rotatedRightDir = vec3(
      particleRightDir.x * cosSpin - particleRightDir.y * sinSpin,
      particleRightDir.x * sinSpin + particleRightDir.y * cosSpin,
      particleRightDir.z
    );
    vec3 rotatedUpDir = vec3(
      particleUpDir.x * cosSpin - particleUpDir.y * sinSpin,
      particleUpDir.x * sinSpin + particleUpDir.y * cosSpin,
      particleUpDir.z
    );
    
    particleRightDir = rotatedRightDir;
    particleUpDir = rotatedUpDir;
  }
  
  return Vat3_ParticleDirection(particleRightDir, particleUpDir);
}

Vat3_PScaleData computePScaleData(vec4 posTexThisFrame, vec4 posTexNextFrame, vec3 boundMax, bool noLerp, bool isRigidBody) {
  float maxDataPScale = 1.0 / (1.0 - fract(boundMax.y));
  float posAlphaThisFrame = posTexThisFrame.a;
  float posAlphaNextFrame = posTexNextFrame.a;
  
  float clampedPScaleThisFrame = isRigidBody ? (1.0 - fract(posAlphaThisFrame * 4.0)) : posAlphaThisFrame;
  float clampedPScaleNextFrame = isRigidBody 
    ? (noLerp ? (1.0 - fract(posAlphaThisFrame * 4.0)) : (1.0 - fract(posAlphaNextFrame * 4.0)))
    : (noLerp ? posAlphaThisFrame : posAlphaNextFrame);
  
  float pScaleThisFrame = clampedPScaleThisFrame * maxDataPScale;
  float pScaleNextFrame = clampedPScaleNextFrame * maxDataPScale;
  
  return Vat3_PScaleData(
    clampedPScaleNextFrame,
    clampedPScaleThisFrame,
    maxDataPScale,
    posAlphaNextFrame,
    posAlphaThisFrame,
    pScaleNextFrame,
    pScaleThisFrame
  );
}

Vat3_ParticleScale computeParticleScale(
  Vat3_PScaleData pScaleData, float particleEnabledThisFrame, float particleEnabledNextFrame, float interpolation
) {
  float particleScaleMultiplier = u_globalParticlePiecesScaleMultiplier * u_additionalParticleScaleUniformMultiplier;
  
  float scaleFactor = u_particlePiecesScaleAreInPositionAlpha 
    ? (u_interframeInterpolation ? mix(pScaleData.posAlphaThisFrame, pScaleData.posAlphaNextFrame, interpolation) : pScaleData.posAlphaThisFrame)
    : 1.0;
  
  float baseScale = scaleFactor * particleScaleMultiplier;
  return Vat3_ParticleScale(baseScale, baseScale);
}

vec4 multiRpfQuaternionSmix(
  float rpf,
  float interpolationAlpha,
  float rotTexAlpha,
  vec4 quaterionThisFrame,
  vec4 quaterionNextFrameInput
) {
  float rpfCycle = fract(rpf) * 0.5;
  float rpfAlphaCycle = fract(rpf * interpolationAlpha) * 0.5;
  float sinRpfAlphaDif = sin((rpfCycle - rpfAlphaCycle) * tau);
  float sinAlpha = sin(rpfAlphaCycle * tau);
  float sinRpf = sin(rpfCycle * tau);

  vec4 quaternionNextFrame = quaterionNextFrameInput;
  if (dot(quaterionThisFrame, quaternionNextFrame) < 0.0) {
    quaternionNextFrame = -quaternionNextFrame;
  }

  vec4 rpfQuaternionThisFrame = sinRpfAlphaDif * quaterionThisFrame;
  vec4 rpfQuaternionNextFrame = sinAlpha * quaternionNextFrame * sign(rotTexAlpha);
  vec4 lerpedQuaternion = (rpfQuaternionThisFrame + rpfQuaternionNextFrame) / sinRpf;

  return normalize(lerpedQuaternion);
}

vec4 computeRigidQuaternion(Vat3_TexturesData textures, float interpolationAlpha) {
  int dominantIndexThisFrame = int(floor(textures.posThisFrame.a * 4.0));
  int dominantIndexNextFrame = int(floor(textures.posNextFrame.a * 4.0));

  vec4 rotThisFrame = textures.rotThisFrame;
  vec4 rotNextFrame = textures.rotNextFrame;

  if (!u_isTexHdr) {
    rotThisFrame = (rotThisFrame - 0.5) * 2.0;
    rotNextFrame = (rotNextFrame - 0.5) * 2.0;
  }

  vec4 quaternionThisFrame = decodeQuaternion(rotThisFrame.xyz, dominantIndexThisFrame);
  vec4 quaternionNextFrame = decodeQuaternion(rotNextFrame.xyz, dominantIndexNextFrame);

  if (u_noLerping || !u_interframeInterpolation) {
    return quaternionThisFrame;
  }

  float rotationsPerFrame = abs(rotThisFrame.w);
  if (rotationsPerFrame <= 0.0001) {
    return quaternionThisFrame;
  }

  float reverseBoundXFract = fract(u_boundMin.x * (-1.0));
  float ldrScale = 1.0 / max(0.0001, 1.0 - reverseBoundXFract);
  float rpf = u_isTexHdr ? rotationsPerFrame : rotationsPerFrame * ldrScale;

  return multiRpfQuaternionSmix(
    rpf,
    interpolationAlpha,
    rotThisFrame.w,
    quaternionThisFrame,
    quaternionNextFrame
  );
}

vec3 getRigidPivot(vec3 position, vec2 uv2, vec2 uv3) {
  vec3 restFramePosition = vec3(uv2.x, uv3.x, uv3.y);
  return position - restFramePosition;
}

// Main deformation router
Vat3_Outputs applyVatDeformation(vec3 position, vec3 normal, vec3 tangent, vec2 uv, vec2 uv1, vec2 uv2, vec2 uv3, int variant) {
  Vat3_Outputs vatOutputs;
  
  Vat3_AnimationData animData = computeAnimationData();
  
  bool shouldInterpolate = !u_noLerping && u_interframeInterpolation;
  
  if (variant == 0) { // Dynamicmesh
    vec2 activeRatio = vec2(1.0); // Dynamic mesh active ratio is 1.0
    Vat3_UVs uvs = computeUVs(uv, animData, u_isTexHdr, activeRatio);
    float interpolationAlpha = shouldInterpolate ? fract(animData.currentFrame) : 0.0;
    
    vec4 lookupThis = vatLoad(u_vatLookupTex, uvs.thisFrameUv);
    vec4 lookupNext = shouldInterpolate ? vatLoad(u_vatLookupTex, uvs.nextFrameUv) : lookupThis;
    
    vec2 decodedThisUv = decodeVatLookupUv(lookupThis, u_isLookupTexHdr);
    vec2 decodedNextUv = shouldInterpolate ? decodeVatLookupUv(lookupNext, u_isLookupTexHdr) : decodedThisUv;
    
    Vat3_TexturesData textures = sampleVat3TexturesFromParams(
      decodedThisUv, decodedNextUv, !shouldInterpolate,
      false, u_usePos2, u_useSpareColor
    );
    
    vec3 boundsRange = u_boundMax - u_boundMin;
    vec3 posThis = decodePosition(textures.posThisFrame, boundsRange, u_boundMin, u_isTexHdr);
    vec3 posNext = decodePosition(textures.posNextFrame, boundsRange, u_boundMin, u_isTexHdr);
    
    if (u_usePos2) {
      posThis += textures.pos2ThisFrame;
      posNext += textures.pos2NextFrame;
    }
    
    vec3 finalPos = shouldInterpolate ? mix(posThis, posNext, interpolationAlpha) : posThis;
    finalPos += u_additionalObjectSpaceOffset;
    
    vec4 colorAndAlpha = interpolateColor(textures.colorThisFrame, textures.colorNextFrame, interpolationAlpha, u_interframeInterpolation, u_interpolateColor);
    vec4 spareColor = interpolateColor(textures.spareColorThisFrame, textures.spareColorNextFrame, interpolationAlpha, u_interframeInterpolation, u_interpolateSpareColor);
    
    Vat3_SurfaceData surface = computeTextureBasedNormals(animData, textures, interpolationAlpha, finalPos);
    vec2 surfaceUv = resolveVatSurfaceUv(colorAndAlpha, textures.posThisFrame.a, u_surfaceUVsfromColorRG);
    
    float animProgress = mix(animData.animProgressThisFrame, animData.animProgressNextFrame, interpolationAlpha);
    
    vatOutputs.outAnimationProgress = vec3(animProgress, animData.animProgressThisFrame, animData.animProgressNextFrame);
    vatOutputs.outPosition = isVatPaddingVertex(uv, u_frameCount) ? vec3(0.0) : finalPos;
    vatOutputs.outNormal = surface.normal;
    vatOutputs.outTangent = surface.tangent;
    vatOutputs.surfaceUv = surfaceUv;
    vatOutputs.outColorAndAlpha = colorAndAlpha;
    vatOutputs.outSpareColorAndAlpha = spareColor;
    
  } else if (variant == 1) { // Softbody
    vec2 activeRatio = computeActivePixelsRatio(u_boundMin, u_boundMax);
    Vat3_UVs uvs = computeUVs(uv1, animData, u_isTexHdr, activeRatio);
    float interpolationAlpha = fract(animData.currentFrame);
    
    Vat3_TexturesData textures = sampleVat3TexturesFromParams(
      uvs.thisFrameUv, uvs.nextFrameUv, u_noLerping,
      false, false, false
    );
    
    vec3 boundsRange = u_boundMax - u_boundMin;
    vec3 posThis = decodePosition(textures.posThisFrame, boundsRange, u_boundMin, u_isTexHdr);
    vec3 posNext = decodePosition(textures.posNextFrame, boundsRange, u_boundMin, u_isTexHdr);
    
    vec3 lerpedPos = interpolateVector3(posThis, posNext, interpolationAlpha, u_interframeInterpolation, true);
    vec3 finalPos = position + lerpedPos + u_additionalObjectSpaceOffset;
    
    vec4 colorAndAlpha = interpolateColor(textures.colorThisFrame, textures.colorNextFrame, interpolationAlpha, u_interframeInterpolation, u_interpolateColor);
    vec4 spareColor = interpolateColor(textures.spareColorThisFrame, textures.spareColorNextFrame, interpolationAlpha, u_interframeInterpolation, u_interpolateSpareColor);
    
    Vat3_SurfaceData surface = computeTextureBasedNormals(animData, textures, interpolationAlpha, finalPos);
    
    float animProgress = mix(animData.animProgressThisFrame, animData.animProgressNextFrame, interpolationAlpha);
    
    vatOutputs.outAnimationProgress = vec3(animProgress, animData.animProgressThisFrame, animData.animProgressNextFrame);
    vatOutputs.outPosition = isVatPaddingVertex(uv1, u_frameCount) ? vec3(0.0) : finalPos;
    vatOutputs.outNormal = surface.normal;
    vatOutputs.outTangent = surface.tangent;
    vatOutputs.surfaceUv = uv;
    vatOutputs.outColorAndAlpha = colorAndAlpha;
    vatOutputs.outSpareColorAndAlpha = spareColor;
    
  } else if (variant == 2) { // Rigidbody
    vec2 activeRatio = computeActivePixelsRatio(u_boundMin, u_boundMax);
    Vat3_UVs uvs = computeUVs(uv1, animData, u_isTexHdr, activeRatio);
    float interpolationAlpha = shouldInterpolate ? fract(animData.currentFrame) : 0.0;
    
    Vat3_TexturesData textures = sampleVat3TexturesFromParams(
      uvs.thisFrameUv, uvs.nextFrameUv, u_noLerping,
      false, u_usePos2, u_useSpareColor
    );
    
    vec3 boundsRange = u_boundMax - u_boundMin;
    vec3 posThis = decodePosition(textures.posThisFrame, boundsRange, u_boundMin, u_isTexHdr);
    vec3 posNext = decodePosition(textures.posNextFrame, boundsRange, u_boundMin, u_isTexHdr);
    
    if (u_usePos2) {
      posThis += textures.pos2ThisFrame;
      posNext += textures.pos2NextFrame;
    }
    
    vec3 piecePosition = shouldInterpolate ? mix(posThis, posNext, interpolationAlpha) : posThis;
    vec4 quaternion = computeRigidQuaternion(textures, interpolationAlpha);
    vec3 restFramePosition = vec3(uv2.x, uv3.x, uv3.y);
    vec3 rotatedPivot = rotateVectorByQuaternion(position - restFramePosition, quaternion);
    
    Vat3_PScaleData pScaleData = computePScaleData(textures.posThisFrame, textures.posNextFrame, u_boundMax, u_noLerping, true);
    float originalPScale = shouldInterpolate ? mix(pScaleData.pScaleThisFrame, pScaleData.pScaleNextFrame, interpolationAlpha) : pScaleData.pScaleThisFrame;
    float finalPScale = u_particlePiecesScaleAreInPositionAlpha ? originalPScale : 1.0;
    
    vec3 stretchScale = vec3(1.0);
    if (u_stretchByVelocity && u_useColorForVelocity) {
      vec4 velocityColor = shouldInterpolate ? mix(textures.colorThisFrame, textures.colorNextFrame, interpolationAlpha) : textures.colorThisFrame;
      vec3 objectSpaceVelocity = vec3(-velocityColor.r, velocityColor.g, velocityColor.b);
      stretchScale = vec3(1.0) + abs(objectSpaceVelocity) * u_stretchByVelocityAmount;
    }
    
    float globalScale = u_globalParticlePiecesScaleMultiplier * finalPScale;
    vec3 pieceVectorWithScale = rotatedPivot * (stretchScale * globalScale);
    vec3 finalPos = pieceVectorWithScale + piecePosition + u_additionalObjectSpaceOffset;
    
    vec4 colorAndAlpha = interpolateColor(textures.colorThisFrame, textures.colorNextFrame, interpolationAlpha, u_interframeInterpolation, u_interpolateColor);
    vec4 spareColor = interpolateColor(textures.spareColorThisFrame, textures.spareColorNextFrame, interpolationAlpha, u_interframeInterpolation, u_interpolateSpareColor);
    
    vec3 rotatedNormal = normalize(rotateVectorByQuaternion(normal, quaternion));
    vec3 rotatedTangent = u_supportSurfaceNormalMaps ? normalize(rotateVectorByQuaternion(tangent, quaternion)) : vec3(0.0);
    
    float animProgress = mix(animData.animProgressThisFrame, animData.animProgressNextFrame, interpolationAlpha);
    
    vatOutputs.outAnimationProgress = vec3(animProgress, animData.animProgressThisFrame, animData.animProgressNextFrame);
    vatOutputs.outPosition = isVatPaddingVertex(uv1, u_frameCount) ? vec3(0.0) : finalPos;
    vatOutputs.outNormal = rotatedNormal;
    vatOutputs.outTangent = rotatedTangent;
    vatOutputs.surfaceUv = uv;
    vatOutputs.outColorAndAlpha = colorAndAlpha;
    vatOutputs.outSpareColorAndAlpha = spareColor;
    
  } else if (variant == 3) { // Particles
    vec2 activeRatio = computeActivePixelsRatio(u_boundMin, u_boundMax);
    Vat3_UVs uvs = computeUVs(uv1, animData, u_isTexHdr, activeRatio);
    float interpolationAlpha = fract(animData.currentFrame);
    
    Vat3_TexturesData textures = sampleVat3TexturesFromParams(
      uvs.thisFrameUv, uvs.nextFrameUv, u_noLerping,
      false, false, true
    );
    
    vec3 boundsRange = u_boundMax - u_boundMin;
    vec3 posThis = decodePosition(textures.posThisFrame, boundsRange, u_boundMin, u_isTexHdr);
    vec3 posNext = decodePosition(textures.posNextFrame, boundsRange, u_boundMin, u_isTexHdr);
    
    Vat3_PScaleData pScaleData = computePScaleData(textures.posThisFrame, textures.posNextFrame, u_boundMax, u_noLerping, false);
    
    vec4 colorAndAlpha = interpolateColor(textures.colorThisFrame, textures.colorNextFrame, interpolationAlpha, u_interframeInterpolation, u_interpolateColor);
    vec4 spareColor = interpolateColor(textures.spareColorThisFrame, textures.spareColorNextFrame, interpolationAlpha, u_interframeInterpolation, u_interpolateSpareColor);
    
    float particleEnabledThisFrame = clamp(sign(length(posThis) - u_originEffectiveRadius), 0.0, 1.0);
    float particleEnabledNextFrame = clamp(sign(length(posNext) - u_originEffectiveRadius), 0.0, 1.0);
    
    Vat3_VelocityData velocityData = calculateVelocity(
      posNext, posThis, colorAndAlpha, u_modelViewMatrix, u_perParticleRandomVelocityScale,
      u_useAlphaForVelocityScale, u_useColorForVelocity, false
    );
    
    Vat3_ParticleDirection particleDirs = calculateParticleOrientation(velocityData);
    Vat3_ParticleScale pScale = computeParticleScale(pScaleData, particleEnabledThisFrame, particleEnabledNextFrame, interpolationAlpha);
    
    vec3 particleRelRightPos = particleDirs.particleRightDir * u_particleWidthBaseScale * pScale.finalScale * (uv.x - 0.5);
    vec3 particleRelUpPos    = particleDirs.particleUpDir * u_particleHeightBaseScale * pScale.finalScale * (uv.y - 0.5);
    vec3 particleFinalPos    = particleRelRightPos + particleRelUpPos + posThis;
    
    vec3 finalPos = particleFinalPos + u_additionalObjectSpaceOffset;
    
    vec3 outNormal = normalize((u_viewToModelMatrix * vec4(0.0, 0.0, 1.0, 0.0)).xyz);
    vec3 outTangent = u_supportSurfaceNormalMaps ? particleDirs.particleRightDir : vec3(0.0);
    
    vec2 particleUvScale = vec2(u_particleTextureUScale, u_particleTextureVScale);
    vec2 particleUvScaleRemaped = particleUvScale * -0.5 + 0.5;
    vec2 surfaceUv = particleUvScaleRemaped + uv * particleUvScale;
    
    float animProgress = mix(animData.animProgressThisFrame, animData.animProgressNextFrame, interpolationAlpha);
    
    vatOutputs.outAnimationProgress = vec3(animProgress, animData.animProgressThisFrame, animData.animProgressNextFrame);
    vatOutputs.outPosition = isVatPaddingVertex(uv1, u_frameCount) ? vec3(0.0) : finalPos;
    vatOutputs.outNormal = outNormal;
    vatOutputs.outTangent = outTangent;
    vatOutputs.surfaceUv = surfaceUv;
    vatOutputs.outColorAndAlpha = colorAndAlpha;
    vatOutputs.outSpareColorAndAlpha = spareColor;
  }

  // ---------- Legacy (pre-VAT3) variants ----------
  // These handle the older Houdini VAT export format used by demo_ examples.
  // Key differences: additive positions (softbody), different UV channels, V-axis always flipped.

  else if (variant == 4) { // Legacy Softbody (demo_cloth)
    // Frame/time calculation: old format uses discrete frame stepping
    float legFrameCount = u_frameCount;
    float legFrame = floor(fract((u_frameRate / (legFrameCount - 0.01)) * VAT_ACTIVE_TIME * u_playbackSpeed) * legFrameCount);
    float legTimeInFrames = mod(legFrame, legFrameCount) * (1.0 / legFrameCount);

    // Reference softVertexShader:
    //   vec4 texturePos = texture(posTexture,vec2(uv2.x, 1.0 - timeInFrames - (1.0 - uv2.y)));
    //   vec4 textureCd = texture(colTexture,vec2(uv2.x, 1.0 - timeInFrames ));
    //   vec4 textureRot = texture(rotTexture,vec2(uv2.x, 1.0 - timeInFrames - (1.0 - uv2.y)));
    // Note: uv2 in reference shader corresponds to Three.js uv1 (second UV channel).
    
    vec2 legSampleUv = vec2(uv1.x, 1.0 - legTimeInFrames - (1.0 - uv1.y));
    vec2 legColUv = vec2(uv1.x, 1.0 - legTimeInFrames);

    vec4 legPosThis = vatLoad(u_vatPosTex, legSampleUv);
    vec4 legRotThis = vatLoad(u_vatRotTex, legSampleUv);
    vec4 legColThis = vatLoad(u_vatColTex, legColUv);

    // Softbody: position is additive offset from rest mesh position
    vec3 legFinalPos = position + legPosThis.xyz;

    // Normal decode: quaternion-style cross product against world up
    vec3 legUp = vec3(0.0, 1.0, 0.0);
    vec3 legNormal = normalize((cross(legRotThis.xyz, cross(legRotThis.xyz, legUp) + (legRotThis.a * legUp)) * 2.0) + legUp);

    vatOutputs.outPosition = legFinalPos;
    vatOutputs.outNormal = legNormal;
    vatOutputs.outTangent = vec3(0.0);
    vatOutputs.surfaceUv = uv;
    vatOutputs.outColorAndAlpha = legColThis;
    vatOutputs.outSpareColorAndAlpha = vec4(0.0);
    vatOutputs.outAnimationProgress = vec3(legTimeInFrames);



  } else if (variant == 5) { // Legacy Rigidbody (demo_rigid_body)
    float legFrameCount = u_frameCount;
    float legFrame = floor(fract((u_frameRate / (legFrameCount - 0.01)) * VAT_ACTIVE_TIME * u_playbackSpeed) * legFrameCount);
    float legTimeInFrames = mod(legFrame, legFrameCount) * (1.0 / legFrameCount);

    // uv1.x is the per-piece X coordinate for texture sampling (FBX UV channel 1)
    // V is 1.0 - legTimeInFrames (Frame 0 is at the bottom of the EXR, V=1.0)
    vec2 legSampleUv = vec2(uv1.x, 1.0 - legTimeInFrames);

    vec4 legPosThis = vatLoad(u_vatPosTex, legSampleUv);
    vec4 legRotThis = vatLoad(u_vatRotTex, legSampleUv);
    vec4 legColThis = vatLoad(u_vatColTex, legSampleUv);

    // Pivot stored in uv2 and uv3 (FBX UV channels 2 & 3)
    // Reference mapping: pivot = vec3(uv3.x, uv4.x, 1.0 - uv4.y)
    //   In Three.js FBXLoader: uv3(ref) = uv2(three), uv4(ref) = uv3(three)
    vec3 legPivot = vec3(uv2.x, uv3.x, 1.0 - uv3.y);
    vec3 legAtOrigin = position - legPivot;

    // Quaternion: maxComponent encoded in pos.w
    int legMaxComponent = int(floor(legPosThis.w * 4.0 + 0.5));
    // Decode quaternion: XYZ from rot texture, W reconstructed
    vec4 legQ = decodeQuaternion(legRotThis.xyz, legMaxComponent);

    // Rotate around pivot then translate to new piece position
    vec3 legRotated = rotateVectorByQuaternion(legAtOrigin, legQ);
    vec3 legFinalPos = legRotated + legPosThis.xyz;

    // Rotate normal and tangent with the same quaternion
    vec3 legNormal = normalize(rotateVectorByQuaternion(normal, legQ));

    vatOutputs.outPosition = legFinalPos;
    vatOutputs.outNormal = legNormal;
    vatOutputs.outTangent = vec3(0.0);
    vatOutputs.surfaceUv = uv;
    vatOutputs.outColorAndAlpha = legColThis;
    vatOutputs.outSpareColorAndAlpha = vec4(0.0);
    vatOutputs.outAnimationProgress = vec3(legTimeInFrames);

  } else if (variant == 6) { // Legacy Fluid / DynamicMesh (demo_fluid)
    float legFrameCount = u_frameCount;
    float legFrame = floor(fract((u_frameRate / (legFrameCount - 0.01)) * VAT_ACTIVE_TIME * u_playbackSpeed) * legFrameCount);
    float legTimeInFrames = mod(legFrame, legFrameCount) * (1.0 / legFrameCount);

    // Fluid FBX lookup UV is in uv (UV0), uv.y values near 1.0.
    // loadRawPngTexture uses flipY=false → V=0 = TOP of PNG file = frame 0.
    // Reference uses TextureLoader (flipY=true) → V=1 = TOP = frame 0,
    //   and samples at V = uv.y - t ≈ 0.9999 (near 1.0 = top).
    // To match: our V = 1.0 - uv.y + t (converts from reference's flipY=true space).
    //   At t=0: V = 1.0 - 0.9999 ≈ 0.0001 (near 0 = TOP in our flipY=false) = frame 0 ✓
    //   As t increases: V increases toward 1.0 (bottom of PNG = last frame) ✓
    float legLookupV = fract(1.0 - uv.y + legTimeInFrames);
    vec2 legLookupUv = vec2(uv.x, legLookupV);
    vec4 legLookupSample = vatLoad(u_vatLookupTex, legLookupUv);

    // Old format lookup decode: g/255+r for U, 1-(a/255+b) for V
    float legDiv = 255.0;
    vec2 legDecodedUv = vec2(
      (legLookupSample.g / legDiv) + legLookupSample.r,
      1.0 - ((legLookupSample.a / legDiv) + legLookupSample.b)
    );

    vec4 legPosThis = vatLoad(u_vatPosTex, legDecodedUv);
    vec4 legRotThis = vatLoad(u_vatRotTex, legDecodedUv);
    vec4 legColThis = vatLoad(u_vatColTex, legDecodedUv);

    // Fluid: position is absolute world-space from texture
    vec3 legFinalPos = legPosThis.xyz;

    // Normal from rot texture (same as legacy softbody)
    vec3 legUp = vec3(0.0, 1.0, 0.0);
    vec3 legNormal = normalize((cross(legRotThis.xyz, cross(legRotThis.xyz, legUp) + (legRotThis.a * legUp)) * 2.0) + legUp);

    vatOutputs.outPosition = legFinalPos;
    vatOutputs.outNormal = legNormal;
    vatOutputs.outTangent = vec3(0.0);
    vatOutputs.surfaceUv = uv;
    vatOutputs.outColorAndAlpha = legColThis;
    vatOutputs.outSpareColorAndAlpha = vec4(0.0);
    vatOutputs.outAnimationProgress = vec3(legTimeInFrames);
  }
  
  return vatOutputs;
}
`;


export class VATEffect {
  constructor(assets, options = {}) {
    this.assets = assets;
    this.mesh = options.instancedMesh || assets.mesh;
    this.type = assets.type;
    this.vatConfig = assets.vatConfig;
    this.isInstanced = Boolean(options.instancedMesh || options.instanced || this.mesh.isInstancedMesh);

    if (this.isInstanced && this.mesh.geometry) {
      const maxCount = this.mesh.instanceMatrix ? this.mesh.instanceMatrix.count : (this.mesh.count || 1);
      const timeOffsets = new Float32Array(maxCount);
      const speedScales = new Float32Array(maxCount);
      speedScales.fill(1.0);

      this.mesh.geometry.setAttribute('vatInstanceTimeOffset', new THREE.InstancedBufferAttribute(timeOffsets, 1));
      this.mesh.geometry.setAttribute('vatInstanceSpeedScale', new THREE.InstancedBufferAttribute(speedScales, 1));
    }

    this._speed = 1.0;
    this._time = 0.0;
    this._enablePlayback = true;
    this._camera = options.camera || null;

    this.uniforms = {
      u_vatColTex: { value: assets.textures.vatColTex },
      u_vatLookupTex: { value: assets.textures.vatLookupTex },
      u_vatPos2Tex: { value: assets.textures.vatPos2Tex },
      u_vatPosTex: { value: assets.textures.vatPosTex },
      u_vatRotTex: { value: assets.textures.vatRotTex },
      u_vatSpareColTex: { value: assets.textures.vatSpareColTex },

      u_time: { value: 0 },
      u_playbackSpeed: { value: 1.0 },
      u_enablePlayback: { value: true },
      u_modelViewMatrix: { value: new THREE.Matrix4() },
      u_viewToModelMatrix: { value: new THREE.Matrix4() },

      u_additionalObjectSpaceOffset: { value: this.vatConfig.staticInputs.additionalObjectSpaceOffset || new THREE.Vector3(0, 0, 0) },
      u_additionalParticleScaleUniformMultiplier: { value: this.vatConfig.staticInputs.additionalParticleScaleUniformMultiplier !== undefined ? this.vatConfig.staticInputs.additionalParticleScaleUniformMultiplier : 1.0 },
      u_animateFirstFrame: { value: Boolean(this.vatConfig.staticInputs.animateFirstFrame) },
      u_boundMax: { value: this.vatConfig.staticInputs.boundMax },
      u_boundMin: { value: this.vatConfig.staticInputs.boundMin },
      u_computeSpinfromHeadingVector: { value: Boolean(this.vatConfig.staticInputs.computeSpinfromHeadingVector) },
      u_displayFrame: { value: this.vatConfig.staticInputs.displayFrame || 0.0 },
      u_frameCount: { value: this.vatConfig.staticInputs.frameCount },
      u_frameRate: { value: this.vatConfig.staticInputs.frameRate },
      u_gameTimeAtFirstFrame: { value: this.vatConfig.staticInputs.gameTimeAtFirstFrame || 0.0 },
      u_globalParticlePiecesScaleMultiplier: { value: this.vatConfig.staticInputs.globalParticlePiecesScaleMultiplier !== undefined ? this.vatConfig.staticInputs.globalParticlePiecesScaleMultiplier : 1.0 },
      u_hideParticlesOverlappingObjectOrigin: { value: Boolean(this.vatConfig.staticInputs.hideParticlesOverlappingObjectOrigin) },
      u_inputTime: { value: 0.0 },
      u_instance: { value: Boolean(this.vatConfig.staticInputs.instance) },
      u_instanceCount: { value: this.vatConfig.staticInputs.instanceCount || 0.0 },
      u_instanceUpdateDynamicData: { value: Boolean(this.vatConfig.staticInputs.instanceUpdateDynamicData) },
      u_interframeInterpolation: { value: this.vatConfig.staticInputs.interframeInterpolation !== undefined ? Boolean(this.vatConfig.staticInputs.interframeInterpolation) : true },
      u_interpolateColor: { value: this.vatConfig.staticInputs.interpolateColor !== undefined ? Boolean(this.vatConfig.staticInputs.interpolateColor) : true },
      u_interpolateSpareColor: { value: this.vatConfig.staticInputs.interpolateSpareColor !== undefined ? Boolean(this.vatConfig.staticInputs.interpolateSpareColor) : true },
      u_isColorTexHdr: { value: Boolean(this.vatConfig.staticInputs.isColorTexHdr) },
      u_isLookupTexHdr: { value: Boolean(this.vatConfig.staticInputs.isLookupTexHdr) },
      u_isTexHdr: { value: Boolean(this.vatConfig.staticInputs.isTexHdr) },
      u_noLerping: { value: Boolean(this.vatConfig.staticInputs.noLerping) },
      u_originEffectiveRadius: { value: this.vatConfig.staticInputs.originEffectiveRadius !== undefined ? this.vatConfig.staticInputs.originEffectiveRadius : 1.0 },
      u_particleHeightBaseScale: { value: this.vatConfig.staticInputs.particleHeightBaseScale !== undefined ? this.vatConfig.staticInputs.particleHeightBaseScale : 0.5 },
      u_particlePiecesScaleAreInPositionAlpha: { value: Boolean(this.vatConfig.staticInputs.particlePiecesScaleAreInPositionAlpha) },
      u_particleShardCount: { value: this.vatConfig.staticInputs.particleShardCount || 0.0 },
      u_particleShardIndex: { value: this.vatConfig.staticInputs.particleShardIndex || 0.0 },
      u_particleShards: { value: Boolean(this.vatConfig.staticInputs.particleShards) },
      u_particleSpinPhase: { value: this.vatConfig.staticInputs.particleSpinPhase || 0.0 },
      u_particleTextureUScale: { value: this.vatConfig.staticInputs.particleTextureUScale !== undefined ? this.vatConfig.staticInputs.particleTextureUScale : 1.0 },
      u_particleTextureVScale: { value: this.vatConfig.staticInputs.particleTextureVScale !== undefined ? this.vatConfig.staticInputs.particleTextureVScale : 1.0 },
      u_particleWidthBaseScale: { value: this.vatConfig.staticInputs.particleWidthBaseScale !== undefined ? this.vatConfig.staticInputs.particleWidthBaseScale : 0.5 },
      u_perParticleRandomSpinSpeed: { value: this.vatConfig.staticInputs.perParticleRandomSpinSpeed || 0.0 },
      u_perParticleRandomVelocityScale: { value: this.vatConfig.staticInputs.perParticleRandomVelocityScale || 0.0 },
      u_scalebyVelocityAmount: { value: this.vatConfig.staticInputs.scalebyVelocityAmount || 0.0 },
      u_spinFromHeading: { value: Boolean(this.vatConfig.staticInputs.spinFromHeading) },
      u_stretchByVelocity: { value: this.vatConfig.staticInputs.stretchByVelocity !== undefined ? Boolean(this.vatConfig.staticInputs.stretchByVelocity) : true },
      u_stretchByVelocityAmount: { value: this.vatConfig.staticInputs.stretchByVelocityAmount !== undefined ? this.vatConfig.staticInputs.stretchByVelocityAmount : 1.0 },
      u_supportSurfaceNormalMaps: { value: this.vatConfig.staticInputs.supportSurfaceNormalMaps !== undefined ? Boolean(this.vatConfig.staticInputs.supportSurfaceNormalMaps) : true },
      u_surfaceNormals: { value: this.vatConfig.staticInputs.surfaceNormals !== undefined ? Boolean(this.vatConfig.staticInputs.surfaceNormals) : true },
      u_surfaceUVsfromColorRG: { value: Boolean(this.vatConfig.staticInputs.surfaceUVsfromColorRG) },
      u_useAlphaForVelocityScale: { value: Boolean(this.vatConfig.staticInputs.useAlphaForVelocityScale) },
      u_useColorForVelocity: { value: Boolean(this.vatConfig.staticInputs.useColorForVelocity) },
      u_useCompressedNormals: { value: Boolean(this.vatConfig.staticInputs.useCompressedNormals) },
      u_useLookup: { value: Boolean(this.vatConfig.staticInputs.useLookup) },
      u_useParticleBillboarding: { value: this.vatConfig.staticInputs.useParticleBillboarding !== undefined ? Boolean(this.vatConfig.staticInputs.useParticleBillboarding) : true },
      u_useParticleVelocitySpin: { value: Boolean(this.vatConfig.staticInputs.useParticleVelocitySpin) },
      u_usePos2: { value: Boolean(this.vatConfig.staticInputs.usePos2) },
      u_useRightHandedCoordinates: { value: Boolean(this.vatConfig.staticInputs.useRightHandedCoordinates) },
      u_useSpareColor: { value: Boolean(this.vatConfig.staticInputs.useSpareColor) },
      u_vertexCount: { value: this.vatConfig.staticInputs.vertexCount }
    };

    // Particles default albedo map if texture is set in options
    this._albedoTexture = options.albedoTexture || null;

    this.initMaterial();
  }

  setCamera(camera) {
    this._camera = camera;
  }

  get speed() {
    return this._speed;
  }

  set speed(val) {
    this._speed = val;
    this.uniforms.u_playbackSpeed.value = val;
  }

  get time() {
    return this._time;
  }

  set time(val) {
    this._time = val;
    this.uniforms.u_inputTime.value = val;
    this.uniforms.u_time.value = val;
  }

  setEnabled(enabled) {
    this._enablePlayback = enabled;
    this.uniforms.u_enablePlayback.value = enabled;
  }

  setInstanceTimeOffset(index, offset) {
    if (!this.isInstanced) return;
    const attr = this.mesh.geometry.getAttribute('vatInstanceTimeOffset');
    if (attr) {
      attr.setX(index, offset);
      attr.needsUpdate = true;
    }
  }

  setInstanceSpeedScale(index, scale) {
    if (!this.isInstanced) return;
    const attr = this.mesh.geometry.getAttribute('vatInstanceSpeedScale');
    if (attr) {
      attr.setX(index, scale);
      attr.needsUpdate = true;
    }
  }

  set albedoTexture(tex) {
    this._albedoTexture = tex;
    if (this.mesh.material) {
      if (Array.isArray(this.mesh.material)) {
        this.mesh.material.forEach(m => {
          if (m.map !== undefined) m.map = tex;
        });
      } else {
        if (this.mesh.material.map !== undefined) this.mesh.material.map = tex;
      }
    }
  }

  initMaterial() {
    const materials = Array.isArray(this.mesh.material) ? this.mesh.material : [this.mesh.material];

    // Variant mapping: Dynamicmesh = 0, Softbody = 1, Rigidbody = 2, Particles = 3
    let variantId = 0;
    if (this.type === 'Softbody') variantId = 1;
    else if (this.type === 'Rigidbody') variantId = 2;
    else if (this.type === 'Particles') variantId = 3;

    // Override with legacy variants for pre-VAT3 demo_ examples
    if (this.vatConfig.staticInputs.legacy) {
      if (this.type === 'Softbody') variantId = 4;         // Legacy Softbody
      else if (this.type === 'Rigidbody') variantId = 5;  // Legacy Rigidbody
      else if (this.type === 'Dynamicmesh') variantId = 6; // Legacy Fluid/DynamicMesh
    }

    materials.forEach((mat, index) => {
      // Clone standard material to apply specific overrides
      const clonedMat = mat.clone();
      clonedMat.side = THREE.DoubleSide;

      // Configure transparency and depth write based on variant type
      if (this.type === 'Particles') {
        clonedMat.transparent = true;
        clonedMat.depthWrite = false;
      } else {
        clonedMat.transparent = false;
        clonedMat.depthWrite = true;
      }

      clonedMat.vertexColors = true; // Always enable vertex colors

      // Force UV and COLOR defines to ensure standard varyings/attributes are declared
      clonedMat.defines = clonedMat.defines || {};
      clonedMat.defines.USE_UV = '';
      clonedMat.defines.USE_UV1 = '';
      clonedMat.defines.USE_UV2 = '';
      clonedMat.defines.USE_UV3 = '';
      if (this.type !== 'Dynamicmesh') {
        clonedMat.defines.USE_COLOR = '';
        if (this.type === 'Particles') {
          clonedMat.defines.USE_COLOR_ALPHA = '';
        }
      }

      if (this.type !== 'Particles' && this._albedoTexture && clonedMat.map !== undefined) {
        clonedMat.map = this._albedoTexture;
      }

      clonedMat.customProgramCacheKey = () => {
        return `vat-${variantId}`;
      };

      clonedMat.onBeforeCompile = (shader) => {
        // 1. Inject uniforms
        Object.keys(this.uniforms).forEach((key) => {
          shader.uniforms[key] = this.uniforms[key];
        });

        // 2. Vertex shader overrides
        let vertex = shader.vertexShader;

        // Determine which attributes this variant needs to declare and pass
        const needsUv1 = variantId === 1 || variantId === 2 || variantId === 3 || variantId === 4 || variantId === 5;
        const needsUv2And3 = variantId === 2 || variantId === 5;

        let declarations = '';
        if (needsUv1) {
          declarations += 'attribute vec2 vatUv1;\n';
        }
        if (needsUv2And3) {
          declarations += 'attribute vec2 vatUv2;\n';
          declarations += 'attribute vec2 vatUv3;\n';
        }

        let instancedDeclarations = '';
        if (this.isInstanced) {
          instancedDeclarations = `
attribute float vatInstanceTimeOffset;
attribute float vatInstanceSpeedScale;
#define VAT_ACTIVE_TIME ((u_inputTime + vatInstanceTimeOffset) * vatInstanceSpeedScale)
\n`;
        }

        // Declare attributes if not present, and common code
        let header = instancedDeclarations + VAT3_GLSL_COMMON + '\n' + declarations + '\n';

        vertex = header + vertex;

        // Replace void main() to compute vatOut immediately at entry point
        const uv1Arg = needsUv1 ? 'vatUv1' : 'vec2(0.0)';
        const uv2Arg = needsUv2And3 ? 'vatUv2' : 'vec2(0.0)';
        const uv3Arg = needsUv2And3 ? 'vatUv3' : 'vec2(0.0)';

        vertex = vertex.replace('void main() {', `
        void main() {
          Vat3_Outputs vatOut = applyVatDeformation(position, normal, vec3(0.0), uv, ${uv1Arg}, ${uv2Arg}, ${uv3Arg}, ${variantId});
        `);





        // Replace normal computation
        const beginNormalInclude = '#include <beginnormal_vertex>';
        const customNormalCode = `
        vec3 objectNormal = vatOut.outNormal;
        #ifdef USE_TANGENT
        vec3 objectTangent = vatOut.outTangent;
        #endif
        `;
        vertex = vertex.replace(beginNormalInclude, customNormalCode);

        // Replace position computation
        const beginVertexInclude = '#include <begin_vertex>';
        const customVertexCode = `
        vec3 transformed = vatOut.outPosition;
        `;
        vertex = vertex.replace(beginVertexInclude, customVertexCode);

        // Inject color mapping in vertex shader if color attribute is present
        const colorVertexInclude = '#include <color_vertex>';
        const customColorCode = `
        #if defined( USE_COLOR_ALPHA )
          vColor = vatOut.outColorAndAlpha;
        #elif defined( USE_COLOR )
          vColor = vatOut.outColorAndAlpha;//.rgb;
        #endif
        `;
        vertex = vertex.replace(colorVertexInclude, customColorCode);

        // Inject vUv override for particles
        if (variantId === 3) {
          const uvVertexInclude = '#include <uv_vertex>';
          vertex = vertex.replace(uvVertexInclude, `
          #include <uv_vertex>
          vUv = vatOut.surfaceUv;
          `);
        }

        shader.vertexShader = vertex;

        // 3. Fragment shader overrides
        let fragment = shader.fragmentShader;

        fragment = `
        uniform bool u_interpolateColor;
        ` + fragment;

        // Add soft radial mask to fragment shader for particles
        if (variantId === 3) {
          fragment = fragment.replace('#include <map_fragment>', `
          #ifdef USE_MAP
            vec4 sampledDiffuseColor = texture2D( map, vUv );
            diffuseColor *= sampledDiffuseColor;
          #endif
          // Soft radial mask for particle transparency
          diffuseColor.a *= smoothstep(0.5, 0.0, length(vUv - vec2(0.5)));
          `);
        }

        shader.fragmentShader = fragment;
      };

      if (Array.isArray(this.mesh.material)) {
        this.mesh.material[index] = clonedMat;
      } else {
        this.mesh.material = clonedMat;
      }
    });

    this.mesh.visible = true;
  }

  update(time) {
    this.time = time;

    if (this._camera) {
      const viewMatrix = this._camera.matrixWorldInverse;
      const worldMatrix = this.mesh.matrixWorld;

      const modelViewMatrix = new THREE.Matrix4();
      modelViewMatrix.multiplyMatrices(viewMatrix, worldMatrix);

      const viewToModelMatrix = new THREE.Matrix4();
      viewToModelMatrix.copy(modelViewMatrix).invert();

      this.uniforms.u_modelViewMatrix.value.copy(modelViewMatrix);
      this.uniforms.u_viewToModelMatrix.value.copy(viewToModelMatrix);
    }
  }

  dispose() {
    const materials = Array.isArray(this.mesh.material) ? this.mesh.material : [this.mesh.material];
    materials.forEach((mat) => mat.dispose());
  }
}
