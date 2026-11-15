// web-ifc needs its WebAssembly file next to the page; copy it from node_modules (cross-platform, no cp)
import { copyFileSync, mkdirSync } from 'node:fs';
mkdirSync('public/wasm', { recursive: true });
copyFileSync('node_modules/web-ifc/web-ifc.wasm', 'public/wasm/web-ifc.wasm');
