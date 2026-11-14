import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import * as WebIFC from 'web-ifc';

const q = new URLSearchParams(location.search);
const url = q.get('m') || '/models/esplanades.ifc';
const stat = document.getElementById('s');
const T = {};
const t0 = performance.now();

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setSize(innerWidth, innerHeight);
document.getElementById('v').appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0xf4f4f2);
const cam = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 5000);
const ctl = new OrbitControls(cam, renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff, 0x888888, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.2); sun.position.set(50, 100, 50); scene.add(sun);
const root = new THREE.Group(); root.rotation.x = -Math.PI / 2; scene.add(root);  // IFC is Z-up

const api = new WebIFC.IfcAPI();
api.SetWasmPath('/wasm/', true);
await api.Init();
T.init = performance.now() - t0;

const buf = new Uint8Array(await (await fetch(url)).arrayBuffer());
T.fetch = performance.now() - t0;
const model = api.OpenModel(buf);
T.open = performance.now() - t0;

const byId = new Map();            // expressID -> [meshes]
const matCache = new Map();
function mat(c) {
  const k = [c.x, c.y, c.z, c.w].map(v => v.toFixed(2)).join();
  if (!matCache.has(k)) matCache.set(k, new THREE.MeshStandardMaterial({
    color: new THREE.Color(c.x, c.y, c.z), opacity: c.w, transparent: c.w < 1,
    side: THREE.DoubleSide }));
  return matCache.get(k);
}
api.StreamAllMeshes(model, (m) => {
  const id = m.expressID;
  for (let i = 0; i < m.geometries.size(); i++) {
    const pg = m.geometries.get(i);
    const g = api.GetGeometry(model, pg.geometryExpressID);
    const v = api.GetVertexArray(g.GetVertexData(), g.GetVertexDataSize());
    const ix = api.GetIndexArray(g.GetIndexData(), g.GetIndexDataSize());
    const bg = new THREE.BufferGeometry();
    const ib = new THREE.InterleavedBuffer(new Float32Array(v), 6);
    bg.setAttribute('position', new THREE.InterleavedBufferAttribute(ib, 3, 0));
    bg.setAttribute('normal', new THREE.InterleavedBufferAttribute(ib, 3, 3));
    bg.setIndex(new THREE.BufferAttribute(new Uint32Array(ix), 1));
    const mesh = new THREE.Mesh(bg, mat(pg.color));
    mesh.applyMatrix4(new THREE.Matrix4().fromArray(pg.flatTransformation));
    mesh.userData.base = mesh.material;
    root.add(mesh);
    if (!byId.has(id)) byId.set(id, []);
    byId.get(id).push(mesh);
    g.delete();
  }
});
T.meshes = performance.now() - t0;

// GUID -> expressID for building elements
const guidToId = new Map();
for (const t of ['IFCDOOR', 'IFCWINDOW', 'IFCSPACE', 'IFCWALL', 'IFCWALLSTANDARDCASE']) {
  const ids = api.GetLineIDsWithType(model, WebIFC[t]);
  for (let i = 0; i < ids.size(); i++)
    guidToId.set(api.GetLine(model, ids.get(i)).GlobalId.value, { id: ids.get(i), t });
}
T.guids = performance.now() - t0;

const HI = new THREE.MeshStandardMaterial({ color: 0xe03030, emissive: 0x501010 });
const DIM = new THREE.MeshStandardMaterial({ color: 0xcccccc, transparent: true, opacity: 0.12, depthWrite: false });
function highlight(guids) {
  const sel = new Set(guids.map(g => guidToId.get(g)?.id).filter(Boolean));
  let n = 0;
  for (const [id, ms] of byId) for (const m of ms) {
    if (!sel.size) m.material = m.userData.base;
    else if (sel.has(id)) { m.material = HI; n++; } else m.material = DIM;
  }
  return n;
}
const guidsOf = (t) => [...guidToId].filter(([, v]) => v.t === t).map(([g]) => g);
document.getElementById('hd').onclick = () => highlight(guidsOf('IFCDOOR'));
document.getElementById('hw').onclick = () => highlight(guidsOf('IFCWINDOW'));
document.getElementById('hs').onclick = () => highlight(guidsOf('IFCSPACE'));
document.getElementById('hc').onclick = () => highlight([]);

const box = new THREE.Box3().setFromObject(root);
const c = box.getCenter(new THREE.Vector3()), sz = box.getSize(new THREE.Vector3()).length();
cam.position.set(c.x + sz * 0.6, c.y + sz * 0.6, c.z + sz * 0.6); ctl.target.copy(c); ctl.update();
(function loop() { requestAnimationFrame(loop); renderer.render(scene, cam); })();

stat.textContent = `${byId.size} objects, ${Object.entries(T).map(([k, v]) => k + ' ' + Math.round(v)).join(', ')} ms`;
window.__spike = { T, objects: byId.size, guids: guidToId.size, highlight, guidsOf, byId };
window.__ready = true;
