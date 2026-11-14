# IFC viewer spike (three + web-ifc)

Checks that an IFC opens in the browser and that elements can be highlighted by GlobalId,
the same id the indexer stores in `evidence`.

    npm i three web-ifc vite
    mkdir -p public/wasm && cp node_modules/web-ifc/*.wasm public/wasm/
    # put a model at public/models/esplanades.ifc, then:
    npx vite          # open /?m=/models/esplanades.ifc

`window.__spike.highlight([guid, ...])` highlights; the rest is dimmed.
Measured on a 12 MB model in headless Chromium (software GL): ~3 s to load and build 1958 objects.
