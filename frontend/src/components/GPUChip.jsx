import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

// Procedural AI-accelerator module, modeled after the classic top-down
// GPU/SoC carrier-board photograph: square green PCB, six metallic heatsink
// shields flanking a silver die-lid, gold edge connector + mounting rings,
// and an iridescent silicon die at the center.
//
// All motion is driven off `loopPhase()` (wall-clock time modulo LOOP_SECONDS)
// rather than each frame's raw elapsed time, so that ANY LOOP_SECONDS-long
// window recorded from this scene tiles into a seamless video loop — this is
// exported to a background <video> on the marketing site rather than
// rendered live (see scripts that generate public/videos/gpu-module-loop.*).
export const LOOP_SECONDS = 8;
const LOOP_HZ = (2 * Math.PI) / LOOP_SECONDS;
const loopPhase = () => (performance.now() / 1000) % LOOP_SECONDS;

const DIE_VERTEX = /* glsl */ `
  varying vec2 vUv;
  varying vec3 vNormalV;
  varying vec3 vViewPos;
  void main() {
    vUv = uv;
    vNormalV = normalize(normalMatrix * normal);
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vViewPos = -mv.xyz;
    gl_Position = projectionMatrix * mv;
  }
`;

const DIE_FRAGMENT = /* glsl */ `
  uniform float uTime;
  uniform float uPulse;
  varying vec2 vUv;
  varying vec3 vNormalV;
  varying vec3 vViewPos;

  float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
  }

  vec3 hsl2rgb(float h, float s, float l) {
    vec3 rgb = clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    float c = (1.0 - abs(2.0 * l - 1.0)) * s;
    return (rgb - 0.5) * c + l;
  }

  void main() {
    vec3 viewDir = normalize(vViewPos);
    vec3 normal = normalize(vNormalV);
    float fresnel = pow(1.0 - clamp(dot(normal, viewDir), 0.0, 1.0), 2.5);

    vec2 macro = floor(vUv * 6.0);
    float baseHue = hash(macro);
    float shift = hash(macro + 11.0) * 0.15;
    float hue = fract(baseHue + shift + 0.05 * sin(uTime) + fresnel * 0.12);
    float light = 0.38 + hash(macro + 3.7) * 0.16;
    vec3 color = hsl2rgb(hue, 0.42, light);

    vec2 fineUv = fract(vUv * 34.0);
    float edge = min(min(fineUv.x, 1.0 - fineUv.x), min(fineUv.y, 1.0 - fineUv.y));
    float grid = smoothstep(0.0, 0.07, edge);
    color *= mix(0.32, 1.0, grid);

    color += vec3(0.9, 0.95, 1.0) * fresnel * 0.35;

    gl_FragColor = vec4(color * uPulse, 1.0);
  }
`;

function Die() {
  const matRef = useRef();
  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uPulse: { value: 1 },
    }),
    []
  );

  useFrame(() => {
    const phase = LOOP_HZ * loopPhase();
    uniforms.uTime.value = phase;
    // idle -> active breathing pulse: tells the static/dynamic story silently
    uniforms.uPulse.value = 0.75 + (0.5 + Math.sin(phase * 2) * 0.5) * 0.7;
  });

  return (
    <mesh position={[0, 0.078, 0]} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[1.08, 1.08, 1, 1]} />
      <shaderMaterial
        ref={matRef}
        vertexShader={DIE_VERTEX}
        fragmentShader={DIE_FRAGMENT}
        uniforms={uniforms}
        toneMapped={false}
      />
    </mesh>
  );
}

function MountRing({ position }) {
  return (
    <group position={position}>
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0.035, 0]}>
        <torusGeometry args={[0.1, 0.028, 12, 24]} />
        <meshStandardMaterial color="#d9c98a" metalness={0.9} roughness={0.25} />
      </mesh>
      <mesh position={[0, 0.03, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <circleGeometry args={[0.075, 20]} />
        <meshStandardMaterial color="#050b08" metalness={0.2} roughness={0.7} />
      </mesh>
    </group>
  );
}

function Heatsink({ position }) {
  return (
    <group position={position}>
      <mesh position={[0, 0.075, 0]}>
        <boxGeometry args={[0.62, 0.15, 0.62]} />
        <meshStandardMaterial color="#a9b4b0" metalness={0.85} roughness={0.28} />
      </mesh>
      {[...Array(5)].map((_, i) => (
        <mesh key={i} position={[-0.24 + i * 0.12, 0.155, 0]}>
          <boxGeometry args={[0.03, 0.02, 0.58]} />
          <meshStandardMaterial color="#c7d0cd" metalness={0.9} roughness={0.2} />
        </mesh>
      ))}
    </group>
  );
}

function Chip({ position, scale = 1 }) {
  return (
    <mesh position={position}>
      <boxGeometry args={[0.16 * scale, 0.045, 0.16 * scale]} />
      <meshStandardMaterial color="#0a0f0c" metalness={0.5} roughness={0.5} />
    </mesh>
  );
}

function TraceLines() {
  const lines = useMemo(() => {
    const items = [];
    for (let i = 0; i < 8; i++) {
      const z = -1.35 + i * 0.06;
      items.push({ x: 0, z, w: 0.9 + Math.random() * 0.4, horizontal: true });
    }
    for (let i = 0; i < 6; i++) {
      const x = -0.9 + i * 0.06;
      items.push({ x, z: 1.1, w: 0.5 + Math.random() * 0.3, horizontal: false });
    }
    return items;
  }, []);

  return (
    <group position={[0, 0.021, 0]}>
      {lines.map((l, i) => (
        <mesh
          key={i}
          position={[l.x, 0, l.z]}
          rotation={l.horizontal ? [0, 0, 0] : [0, Math.PI / 2, 0]}
        >
          <boxGeometry args={[l.w, 0.004, 0.012]} />
          <meshStandardMaterial
            color="#0a3322"
            emissive="#22c55e"
            emissiveIntensity={0.5}
            metalness={0.6}
            roughness={0.4}
          />
        </mesh>
      ))}
    </group>
  );
}

function AIAcceleratorModule() {
  const heatsinkZ = [-1.05, 0, 1.05];

  return (
    <group>
      {/* PCB substrate */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[3.2, 0.09, 3.2]} />
        <meshStandardMaterial color="#0e3d28" metalness={0.35} roughness={0.55} />
      </mesh>

      {/* Gold perimeter trace border */}
      <mesh position={[0, 0.048, 0]} rotation={[-Math.PI / 2, 0, Math.PI / 4]}>
        <ringGeometry args={[1.5, 1.56, 4, 1]} />
        <meshStandardMaterial color="#d9c98a" metalness={0.85} roughness={0.3} side={THREE.DoubleSide} />
      </mesh>

      <TraceLines />

      {/* Corner mounting rings */}
      <MountRing position={[-1.4, 0, -1.4]} />
      <MountRing position={[1.4, 0, -1.4]} />
      <MountRing position={[-1.4, 0, 1.4]} />
      <MountRing position={[1.4, 0, 1.4]} />

      {/* Heatsink columns */}
      {heatsinkZ.map((z) => (
        <group key={`l-${z}`}>
          <Heatsink position={[-1.15, 0, z]} />
          <Chip position={[-0.72, 0.02, z]} />
        </group>
      ))}
      {heatsinkZ.map((z) => (
        <group key={`r-${z}`}>
          <Heatsink position={[1.15, 0, z]} />
          <Chip position={[0.72, 0.02, z]} />
        </group>
      ))}

      {/* Central die carrier */}
      <mesh position={[0, 0.035, 0]}>
        <boxGeometry args={[1.5, 0.07, 1.5]} />
        <meshStandardMaterial color="#c9d2cf" metalness={0.75} roughness={0.3} />
      </mesh>
      <mesh position={[0, 0.048, 0]}>
        <boxGeometry args={[1.24, 0.045, 1.24]} />
        <meshStandardMaterial color="#173d3a" metalness={0.5} roughness={0.4} />
      </mesh>
      <Die />

      {/* Small SMD components, top and bottom clusters */}
      {[...Array(6)].map((_, i) => (
        <Chip
          key={`top-${i}`}
          position={[-0.36 + (i % 3) * 0.36, 0.02, -1.42 + Math.floor(i / 3) * 0.16]}
        />
      ))}
      {[...Array(4)].map((_, i) => (
        <Chip
          key={`bot-${i}`}
          position={[-0.18 + (i % 2) * 0.36, 0.02, 1.28 + Math.floor(i / 2) * 0.16]}
          scale={1.2}
        />
      ))}

      {/* Top connector bar */}
      <mesh position={[0, 0.06, -1.5]}>
        <boxGeometry args={[1.4, 0.06, 0.14]} />
        <meshStandardMaterial color="#9aa39f" metalness={0.4} roughness={0.5} />
      </mesh>

      {/* Bottom gold edge connector */}
      <mesh position={[-0.1, 0.055, 1.52]}>
        <boxGeometry args={[2.3, 0.05, 0.1]} />
        <meshStandardMaterial color="#d9c98a" metalness={0.9} roughness={0.2} />
      </mesh>
      <mesh position={[1.35, 0.055, 1.52]} rotation={[0, Math.PI / 4, 0]}>
        <boxGeometry args={[0.22, 0.05, 0.1]} />
        <meshStandardMaterial color="#ff7a00" metalness={0.3} roughness={0.4} emissive="#ff7a00" emissiveIntensity={0.4} toneMapped={false} />
      </mesh>
    </group>
  );
}

// Deterministic pseudo-random in [0,1) — same seed always gives the same
// value, which is what makes the particle field exactly loopable: each
// particle's path is a pure function of (index, loop phase), not accumulated
// velocity state.
function seeded(i) {
  const s = Math.sin(i * 12.9898) * 43758.5453;
  return s - Math.floor(s);
}

function DataFlowParticles() {
  const pointsRef = useRef();
  const count = 200;

  const [positions, seeds] = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const seedArr = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
      seedArr[i * 4] = seeded(i * 3 + 1); // x
      seedArr[i * 4 + 1] = seeded(i * 3 + 2); // z
      seedArr[i * 4 + 2] = seeded(i * 3 + 3); // phase offset
      seedArr[i * 4 + 3] = 1 + Math.floor(seeded(i * 3 + 4) * 3); // integer cycles per loop (for a perfect seam)
    }
    return [pos, seedArr];
  }, []);

  useFrame(() => {
    if (!pointsRef.current) return;
    const posArr = pointsRef.current.geometry.attributes.position.array;
    const loopFrac = loopPhase() / LOOP_SECONDS;
    for (let i = 0; i < count; i++) {
      const sx = seeds[i * 4];
      const sz = seeds[i * 4 + 1];
      const offset = seeds[i * 4 + 2];
      const cycles = seeds[i * 4 + 3];
      const rise = (loopFrac * cycles + offset) % 1;
      posArr[i * 3] = (sx - 0.5) * 4;
      posArr[i * 3 + 1] = -0.5 + rise * 3;
      posArr[i * 3 + 2] = (sz - 0.5) * 4;
    }
    pointsRef.current.geometry.attributes.position.needsUpdate = true;
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
          count={count}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.02}
        color="#3ddc84"
        transparent
        opacity={0.5}
        sizeAttenuation
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

export default function GPUChip() {
  const groupRef = useRef();

  useFrame(() => {
    if (groupRef.current) {
      const phase = LOOP_HZ * loopPhase();
      groupRef.current.position.y = Math.sin(phase) * 0.06;
      groupRef.current.rotation.y = Math.sin(phase) * 0.18;
    }
  });

  return (
    <group ref={groupRef} scale={1.05}>
      <AIAcceleratorModule />
      <DataFlowParticles />

      <pointLight position={[2, 3, 2]} intensity={1.1} color="#FF9500" />
      <pointLight position={[-2, 1.5, -1]} intensity={0.6} color="#2ee6a6" />
      <pointLight position={[0, 2.5, 0]} intensity={0.5} color="#ffffff" />
    </group>
  );
}
