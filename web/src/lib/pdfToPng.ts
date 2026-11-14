/**
 * A PDF without a text layer (a scan, a picture of a drawing) is not vector data: it is treated as an image.
 * The browser draws the first page with pdf.js and hands it back as a PNG, so the server needs no PDF renderer.
 */
export async function pdfFirstPageToPng(file: File, maxSide = 3200): Promise<{ png: File; pages: number; width: number; height: number }> {
  const pdfjs = await import('pdfjs-dist');
  const worker = (await import('pdfjs-dist/build/pdf.worker.min.mjs?url')).default;
  pdfjs.GlobalWorkerOptions.workerSrc = worker;
  const loading = pdfjs.getDocument({ data: new Uint8Array(await file.arrayBuffer()) });
  try {
    const doc = await loading.promise;
    const page = await doc.getPage(1);
    const base = page.getViewport({ scale: 1 });
    const scale = Math.min(maxSide / Math.max(base.width, base.height), 4);   // sharp enough to read, small enough to upload
    const vp = page.getViewport({ scale });
    const cv = document.createElement('canvas');
    cv.width = Math.round(vp.width);
    cv.height = Math.round(vp.height);
    const ctx = cv.getContext('2d')!;
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, cv.width, cv.height);                                  // PDF pages are transparent: PNG needs a paper colour
    await page.render({ canvasContext: ctx, viewport: vp } as never).promise;
    const blob = await new Promise<Blob>((res, rej) => cv.toBlob(b => (b ? res(b) : rej(new Error('PNG encoding failed'))), 'image/png'));
    const name = file.name.replace(/\.pdf$/i, '') + '-page1.png';
    return { png: new File([blob], name, { type: 'image/png' }), pages: doc.numPages, width: cv.width, height: cv.height };
  } finally {
    loading.destroy();
  }
}
