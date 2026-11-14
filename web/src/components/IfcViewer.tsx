import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import * as WebIFC from 'web-ifc';
import type { Evidence } from '../lib/api';
import { Card, CardBody, CardHeader, CardTitle, cn } from './ui';

const FOUND = 0xe11d48;
const MAX_BADGES = 99;

type Scene3D = {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  root: THREE.Group;
  badges: THREE.Group;
  clip: THREE.Plane;
  bounds: THREE.Box3;
  byId: Map<number, THREE.Mesh[]>;
  spaces: Set<number>;
  guidToId: Map<string, number> | null;
  ghost: THREE.MeshStandardMaterial | null;
  api: WebIFC.IfcAPI;
  model: number;
  dirty: boolean;
  dispose: () => void;
};

const badgeTexture = (n: number) => {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d')!;
  g.fillStyle = '#e11d48';
  g.beginPath(); g.arc(32, 32, 29, 0, Math.PI * 2); g.fill();
  g.lineWidth = 4; g.strokeStyle = '#fff'; g.stroke();
  g.fillStyle = '#fff'; g.font = `bold ${n > 9 ? 28 : 34}px system-ui, sans-serif`;
  g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(String(n), 32, 34);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
};

/**
 * The real model in the browser: the IFC is parsed by web-ifc (WebAssembly) and drawn with three.js.
 * What the question is about is red and numbered like the evidence table, the rest of the building is
 * ghosted. "Cut" removes everything above a height, which with the top view gives a true floor plan.
 */
export function IfcViewer({ drawingId, evidence }: { drawingId: string; evidence: Evidence[] }) {
  const host = useRef<HTMLDivElement>(null);
  const S = useRef<Scene3D | null>(null);
  const [phase, setPhase] = useState<'download' | 'parse' | 'ready' | 'error'>('download');
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [cut, setCut] = useState(1);          // 0..1 of the model height; 1 = no cut
  const [shown, setShown] = useState<number | null>(null);

  const found = useMemo(() => evidence.filter(e => e.ref && e.ref.length === 22), [evidence]);

  /* ----------------------------------------------------------- load the model */
  useEffect(() => {
    const el = host.current;
    if (!el) return;
    let dead = false;
    setPhase('download'); setProgress(0); setError(null); setShown(null); setCut(1);

    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.localClippingEnabled = true;
    el.appendChild(renderer.domElement);
    renderer.domElement.style.display = 'block';
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf4f5f7);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f98, 1.7));
    const sun = new THREE.DirectionalLight(0xffffff, 1.3);
    sun.position.set(60, 120, 40);
    scene.add(sun);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 20000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = false;
    const root = new THREE.Group();
    // web-ifc already returns Y-up geometry: rotating it again lays the building on its side
    scene.add(root);
    const badges = new THREE.Group();
    scene.add(badges);
    const clip = new THREE.Plane(new THREE.Vector3(0, -1, 0), 1e9);

    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight;
      renderer.setSize(w, h, false);
      renderer.domElement.style.width = '100%';
      renderer.domElement.style.height = '100%';
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
      if (S.current) S.current.dirty = true;
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();

    const api = new WebIFC.IfcAPI();
    const st: Scene3D = {
      renderer, scene, camera, controls, root, badges, clip, bounds: new THREE.Box3(),
      byId: new Map(), spaces: new Set(), guidToId: null, ghost: null, api, model: -1, dirty: true,
      dispose: () => {},
    };
    controls.addEventListener('change', () => { st.dirty = true; });
    let raf = 0;
    const loop = () => {
      raf = requestAnimationFrame(loop);
      if (st.dirty) { st.dirty = false; renderer.render(scene, camera); }
    };
    loop();
    st.dispose = () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      try { if (st.model >= 0) api.CloseModel(st.model); } catch { /* already closed */ }
      root.traverse(o => { const m = o as THREE.Mesh; if (m.geometry) m.geometry.dispose(); });
      renderer.dispose();
      renderer.domElement.remove();
    };

    (async () => {
      try {
        api.SetWasmPath('/wasm/', true);
        await api.Init();

        // download with progress
        const res = await fetch(`/api/drawings/${drawingId}/source`);
        if (!res.ok) throw new Error(((await res.json().catch(() => null)) as { error?: string } | null)?.error ?? `${res.status} ${res.statusText}`);
        const total = Number(res.headers.get('content-length')) || 0;
        const reader = res.body!.getReader();
        const chunks: Uint8Array[] = [];
        let got = 0;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value); got += value.length;
          if (total) setProgress(got / total);
          if (dead) return;
        }
        const data = new Uint8Array(got);
        let off = 0;
        for (const c of chunks) { data.set(c, off); off += c.length; }
        if (dead) return;

        setPhase('parse');
        await new Promise(r => setTimeout(r, 30));           // let the "parsing" label paint before the long synchronous work
        const model = api.OpenModel(data, { COORDINATE_TO_ORIGIN: true });
        st.model = model;

        const mats = new Map<string, THREE.MeshStandardMaterial>();
        const material = (c: { x: number; y: number; z: number; w: number }) => {
          const k = [c.x, c.y, c.z, c.w].map(v => v.toFixed(2)).join();
          let m = mats.get(k);
          if (!m) {
            m = new THREE.MeshStandardMaterial({
              color: new THREE.Color(c.x, c.y, c.z), opacity: c.w, transparent: c.w < 1, side: THREE.DoubleSide,
              clippingPlanes: [clip],
            });
            mats.set(k, m);
          }
          return m;
        };

        api.StreamAllMeshes(model, mesh => {
          const id = mesh.expressID;
          for (let i = 0; i < mesh.geometries.size(); i++) {
            const pg = mesh.geometries.get(i);
            const g = api.GetGeometry(model, pg.geometryExpressID);
            const v = api.GetVertexArray(g.GetVertexData(), g.GetVertexDataSize());
            const ix = api.GetIndexArray(g.GetIndexData(), g.GetIndexDataSize());
            const bg = new THREE.BufferGeometry();
            const ib = new THREE.InterleavedBuffer(new Float32Array(v), 6);
            bg.setAttribute('position', new THREE.InterleavedBufferAttribute(ib, 3, 0));
            bg.setAttribute('normal', new THREE.InterleavedBufferAttribute(ib, 3, 3));
            bg.setIndex(new THREE.BufferAttribute(new Uint32Array(ix), 1));
            const m = new THREE.Mesh(bg, material(pg.color));
            m.applyMatrix4(new THREE.Matrix4().fromArray(pg.flatTransformation));
            m.userData.base = m.material;
            root.add(m);
            const list = st.byId.get(id) ?? [];
            list.push(m);
            st.byId.set(id, list);
            g.delete();
          }
        });

        // spaces are volumes, not things you see: shown only when a question is about them
        const sp = api.GetLineIDsWithType(model, WebIFC.IFCSPACE);
        for (let i = 0; i < sp.size(); i++) st.spaces.add(sp.get(i));
        for (const id of st.spaces) st.byId.get(id)?.forEach(m => { m.visible = false; });

        root.updateMatrixWorld(true);
        st.bounds = new THREE.Box3().setFromObject(root);
        S.current = st;
        fit(st, null);
        setPhase('ready');
      } catch (e) {
        if (!dead) { setError(e instanceof Error ? e.message : String(e)); setPhase('error'); }
      }
    })();

    return () => {
      dead = true;
      S.current = null;
      st.dispose();
    };
  }, [drawingId]);

  /* ------------------------------------------------------ mark what was found */
  useEffect(() => {
    const st = S.current;
    if (!st || phase !== 'ready') return;
    const { api, model, byId, root, badges } = st;

    if (!st.guidToId) {
      // GlobalId -> express id, once; the evidence refers to objects by GlobalId
      const map = new Map<string, number>();
      for (const id of byId.keys()) {
        try {
          const line = api.GetLine(model, id, false);
          const g = line?.GlobalId?.value;
          if (g) map.set(g, id);
        } catch { /* not a product */ }
      }
      st.guidToId = map;
    }

    badges.clear();
    const ids = new Map<number, number>();                   // express id -> number in the table
    found.forEach((e, i) => { const id = st.guidToId!.get(e.ref!); if (id != null && !ids.has(id)) ids.set(id, i + 1); });

    const ghost = new THREE.MeshStandardMaterial({ color: 0xa4adbb, transparent: true, opacity: 0.14, depthWrite: false, side: THREE.DoubleSide, clippingPlanes: [st.clip] });
    st.ghost = ghost;
    const hit = new THREE.MeshStandardMaterial({ color: FOUND, emissive: 0x6b0f22, side: THREE.DoubleSide, clippingPlanes: [st.clip] });
    const any = ids.size > 0;
    for (const [id, meshes] of byId) {
      const isSpace = st.spaces.has(id);
      for (const m of meshes) {
        if (ids.has(id)) { m.visible = true; m.material = hit; }
        else if (isSpace) m.visible = false;
        else { m.visible = true; m.material = any ? ghost : (m.userData.base as THREE.Material); }
      }
    }

    // numbered badges at the centre of each found object
    const size = st.bounds.getSize(new THREE.Vector3()).length();
    if (ids.size <= MAX_BADGES) {
      for (const [id, n] of ids) {
        const box = new THREE.Box3();
        byId.get(id)?.forEach(m => box.expandByObject(m));
        if (box.isEmpty()) continue;
        const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: badgeTexture(n), depthTest: false, sizeAttenuation: true }));
        sprite.position.copy(box.getCenter(new THREE.Vector3()));
        sprite.scale.setScalar(size * 0.018);
        sprite.renderOrder = 10;
        badges.add(sprite);
      }
    }
    root.updateMatrixWorld(true);
    setShown(any ? ids.size : null);
    fit(st, any ? ids : null);
    // like an architectural cut: remove what is above the found objects so the storey can be seen
    if (any) {
      const fb = new THREE.Box3();
      for (const id of ids.keys()) byId.get(id)?.forEach(m => fb.expandByObject(m));
      const h = st.bounds.max.y - st.bounds.min.y;
      setCut(h > 0 ? Math.min(1, Math.max(0.05, (fb.max.y + 0.1 - st.bounds.min.y) / h)) : 1);
    } else setCut(1);
    st.dirty = true;
  }, [found, phase]);

  /* ----------------------------------------------------------------- the cut */
  useEffect(() => {
    const st = S.current;
    if (!st || phase !== 'ready') return;
    const { min, max } = st.bounds;
    st.clip.constant = cut >= 0.999 ? 1e9 : min.y + (max.y - min.y) * cut;
    // a cut model is read like a plan: the rest is denser so walls and rooms are recognisable
    if (st.ghost) st.ghost.opacity = cut >= 0.999 ? 0.14 : 0.5;
    st.dirty = true;
  }, [cut, phase]);

  function view(mode: 'iso' | 'top') {
    const st = S.current;
    if (!st) return;
    const box = st.bounds;
    const c = box.getCenter(new THREE.Vector3());
    const d = box.getSize(new THREE.Vector3()).length();
    if (mode === 'top') st.camera.position.set(c.x, c.y + d * 0.9, c.z + 0.01);
    else st.camera.position.set(c.x + d * 0.55, c.y + d * 0.5, c.z + d * 0.55);
    st.controls.target.copy(c);
    st.controls.update();
    st.dirty = true;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{shown ? `Found: ${shown} object${shown === 1 ? '' : 's'} in the 3D model` : '3D model'}</CardTitle>
        <span className="flex items-center gap-1 text-xs">
          <button onClick={() => view('iso')} className="rounded-md border border-slate-300 px-2 py-0.5 hover:bg-slate-100">3D</button>
          <button onClick={() => view('top')} className="rounded-md border border-slate-300 px-2 py-0.5 hover:bg-slate-100">Top view</button>
        </span>
      </CardHeader>
      <CardBody className="space-y-3">
        <div className="relative">
          <div ref={host} className="h-[68vh] min-h-[22rem] w-full overflow-hidden rounded-md border border-slate-200" role="img"
            aria-label="3D model; drag to rotate, scroll to zoom, right-drag to pan" />
          {phase !== 'ready' && (
            <div className={cn('absolute inset-0 flex flex-col items-center justify-center gap-2 rounded-md bg-white/85 text-sm', phase === 'error' && 'text-rose-800')}>
              {phase === 'download' && (
                <>
                  <span>Loading the model… {progress ? `${Math.round(progress * 100)}%` : ''}</span>
                  <div className="h-2 w-56 overflow-hidden rounded bg-slate-200"><div className="h-full bg-slate-900" style={{ width: `${Math.round(progress * 100)}%` }} /></div>
                </>
              )}
              {phase === 'parse' && <span>Building the 3D geometry in your browser… large models take several seconds.</span>}
              {phase === 'error' && <span role="alert">The 3D view is not available: {error}</span>}
            </div>
          )}
        </div>
        <label className="flex items-center gap-3 text-xs text-slate-600">
          <span className="w-24 shrink-0">Cut at height</span>
          <input type="range" min={0.02} max={1} step={0.01} value={cut} onChange={e => setCut(Number(e.target.value))} className="w-full" disabled={phase !== 'ready'} />
          <span className="w-16 text-right tabular-nums">{cut >= 0.999 ? 'whole' : `${Math.round(cut * 100)}%`}</span>
        </label>
        <p className="text-xs text-slate-500">
          Drag to rotate, scroll to zoom, right-drag to pan. Lower the cut and switch to <b>Top view</b> to read a floor plan straight from the model.
          {found.length > 0 && <span className="text-rose-700"> Red and numbered = what you asked about; the rest of the building is ghosted.</span>}
        </p>
      </CardBody>
    </Card>
  );
}

/** Point the camera at the found objects, or at the whole model when nothing is marked. */
function fit(st: Scene3D, ids: Map<number, number> | null) {
  const box = new THREE.Box3();
  if (ids && ids.size) for (const id of ids.keys()) st.byId.get(id)?.forEach(m => box.expandByObject(m));
  if (box.isEmpty()) box.copy(st.bounds);
  const c = box.getCenter(new THREE.Vector3());
  // never closer than a third of the building: a single door must still be seen in its surroundings
  const d = Math.max(box.getSize(new THREE.Vector3()).length(), st.bounds.getSize(new THREE.Vector3()).length() * (ids ? 0.45 : 0), 1);
  // far enough that the whole bounding sphere fits the view, with a margin
  const dist = (d / 2) / Math.sin(THREE.MathUtils.degToRad(st.camera.fov / 2)) * (ids ? 1.1 : 0.8);
  const dir = ids ? new THREE.Vector3(0.15, 0.95, 0.45) : new THREE.Vector3(0.55, 0.5, 0.65);
  dir.normalize();
  st.camera.position.set(c.x + dir.x * dist, c.y + dir.y * dist, c.z + dir.z * dist);
  st.controls.target.copy(c);
  st.camera.near = Math.max(d / 1000, 0.05);
  st.camera.far = Math.max(d * 40, 2000);
  st.camera.updateProjectionMatrix();
  st.controls.update();
  st.dirty = true;
}
