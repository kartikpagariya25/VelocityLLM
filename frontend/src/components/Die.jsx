import { useMemo, useRef } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'

const N = 14
const HOT = new THREE.Color('#ff6a00')
const GOLD = new THREE.Color('#ffc34d')
const EMBER = new THREE.Color('#3a1d0a')

function Cores() {
  const mesh = useRef()
  const dummy = useMemo(() => new THREE.Object3D(), [])
  const tint = useMemo(() => new THREE.Color(), [])
  const cells = useMemo(() => {
    const out = []
    for (let i = 0; i < N; i += 1) for (let j = 0; j < N; j += 1) out.push([i - (N - 1) / 2, j - (N - 1) / 2])
    return out
  }, [])

  useFrame(({ clock }) => {
    const t = clock.elapsedTime
    cells.forEach(([x, z], i) => {
      const wave = Math.sin(x * 0.7 + t * 1.3) * Math.cos(z * 0.7 - t * 0.9)
      const heat = Math.max(0, wave)
      const h = 0.14 + heat * 0.7
      dummy.position.set(x, h / 2, z)
      dummy.scale.set(0.74, h, 0.74)
      dummy.updateMatrix()
      mesh.current.setMatrixAt(i, dummy.matrix)
      tint.copy(EMBER).lerp(HOT, Math.min(1, heat * 1.4)).lerp(GOLD, Math.max(0, heat - 0.75) * 3)
      mesh.current.setColorAt(i, tint)
    })
    mesh.current.instanceMatrix.needsUpdate = true
    mesh.current.instanceColor.needsUpdate = true
  })

  return (
    <instancedMesh ref={mesh} args={[undefined, undefined, cells.length]}>
      <boxGeometry args={[1, 1, 1]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  )
}

function Rig({ children }) {
  const group = useRef()
  const wide = useThree((state) => state.size.width / state.size.height > 1)
  useFrame(({ pointer }) => {
    const y = typeof window === 'undefined' ? 0 : window.scrollY / 900
    group.current.rotation.y += (pointer.x * 0.35 + 0.6 + y - group.current.rotation.y) * 0.05
    group.current.rotation.x += (0.85 - pointer.y * 0.12 - group.current.rotation.x) * 0.05
  })
  return (
    <group ref={group} position={wide ? [5.2, 0, -2] : [0, 3, -4]} scale={wide ? 0.78 : 0.62}>
      {children}
    </group>
  )
}

export default function Die({ active, still }) {
  return (
    <Canvas
      dpr={[1, 1.75]}
      camera={{ position: [0, 9, 16], fov: 38 }}
      frameloop={still ? 'demand' : active ? 'always' : 'never'}
      gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
    >
      <Rig>
        <mesh position={[0, -0.12, 0]}>
          <boxGeometry args={[N + 1.2, 0.2, N + 1.2]} />
          <meshBasicMaterial color="#1c130c" />
        </mesh>
        <mesh position={[0, -0.23, 0]}>
          <boxGeometry args={[N + 2, 0.05, N + 2]} />
          <meshBasicMaterial color="#ff6a00" transparent opacity={0.25} />
        </mesh>
        <Cores />
      </Rig>
    </Canvas>
  )
}
