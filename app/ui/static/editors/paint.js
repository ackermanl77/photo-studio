// Selection painter (app.ui_editors._paint_editor_html): brush / radial / gradient painted as white
// at varying alpha on one canvas -- alpha is the mask value, so the commit is the canvas as a PNG.
(function() {
  const maskCanvas = document.getElementById('mask-canvas');
  const previewCanvas = document.getElementById('preview-canvas');
  const ctx = maskCanvas.getContext('2d');
  const pctx = previewCanvas.getContext('2d');
  const W = maskCanvas.width, H = maskCanvas.height;
  let tool = 'brush', erasing = false, brushSize = 18;
  let drawing = false, lastX = 0, lastY = 0, startX = 0, startY = 0;
  if (CONFIG.preloadMaskB64) {
    // Re-editing an existing painted selection: continue from what was there.
    const preload = new Image();
    preload.onload = () => ctx.drawImage(preload, 0, 0, W, H);
    preload.src = 'data:image/png;base64,' + CONFIG.preloadMaskB64;
  }

  document.querySelectorAll('.tool-btn[data-tool]').forEach((btn) => {
    btn.addEventListener('click', () => {
      tool = btn.dataset.tool;
      document.querySelectorAll('.tool-btn[data-tool]').forEach((b) => b.classList.toggle('active', b === btn));
    });
  });
  document.getElementById('erase-btn').addEventListener('click', (e) => {
    erasing = !erasing;
    e.currentTarget.classList.toggle('active', erasing);
  });
  document.getElementById('clear-btn').addEventListener('click', () => { ctx.clearRect(0, 0, W, H); lastShape = null; pctx.clearRect(0, 0, W, H); });
  const sizeInput = document.getElementById('brush-size');
  const sizeLabel = document.getElementById('brush-size-label');
  function setBrushSize(value) {
    brushSize = Math.max(Number(sizeInput.min), Math.min(Number(sizeInput.max), value));
    sizeInput.value = brushSize;
    sizeLabel.textContent = brushSize;
  }
  sizeInput.addEventListener('input', (e) => setBrushSize(Number(e.target.value)));

  // The last radial/gradient stays adjustable from the keyboard until the next stroke (F11):
  // the mask as it was before that shape is kept, so a nudge repaints the shape moved.
  let lastShape = null, beforeShape = null;
  function outlineShape() {
    pctx.clearRect(0, 0, W, H);
    if (!lastShape) return;
    pctx.strokeStyle = 'rgba(251,191,36,0.95)'; pctx.lineWidth = 1.5; pctx.setLineDash([4, 3]);
    pctx.beginPath();
    if (lastShape.tool === 'radial') {
      pctx.arc(lastShape.x0, lastShape.y0, Math.hypot(lastShape.x1 - lastShape.x0, lastShape.y1 - lastShape.y0), 0, Math.PI * 2);
    } else {
      pctx.moveTo(lastShape.x0, lastShape.y0); pctx.lineTo(lastShape.x1, lastShape.y1);
    }
    pctx.stroke();
  }
  function repaintShape() {
    ctx.putImageData(lastShape.before, 0, 0);
    const wasErasing = erasing;
    erasing = lastShape.erasing;
    if (lastShape.tool === 'radial') paintRadial(lastShape.x0, lastShape.y0, Math.hypot(lastShape.x1 - lastShape.x0, lastShape.y1 - lastShape.y0));
    else paintGradient(lastShape.x0, lastShape.y0, lastShape.x1, lastShape.y1);
    erasing = wasErasing;
    outlineShape();
  }
  const ARROWS = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]};
  document.addEventListener('keydown', (e) => {
    if (e.target && e.target.tagName === 'INPUT') return;
    if (e.key === '[' || e.key === ']') {
      e.preventDefault();
      setBrushSize(brushSize + (e.key === ']' ? 1 : -1) * (e.shiftKey ? 10 : 2));
      return;
    }
    const move = ARROWS[e.key];
    if (!move || !lastShape) return;
    e.preventDefault();
    const step = e.shiftKey ? 10 : 1;
    if (!e.altKey) { lastShape.x0 += move[0] * step; lastShape.y0 += move[1] * step; }
    lastShape.x1 += move[0] * step; lastShape.y1 += move[1] * step;
    repaintShape();
  });

  function toCanvasXY(e) {
    const rect = maskCanvas.getBoundingClientRect();
    return [(e.clientX - rect.left) * (W / rect.width), (e.clientY - rect.top) * (H / rect.height)];
  }

  function paintDab(x, y) {
    ctx.globalCompositeOperation = erasing ? 'destination-out' : 'source-over';
    ctx.strokeStyle = 'white'; ctx.fillStyle = 'white';
    ctx.lineWidth = brushSize * 2; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.beginPath(); ctx.moveTo(lastX, lastY); ctx.lineTo(x, y); ctx.stroke();
    ctx.beginPath(); ctx.arc(x, y, brushSize, 0, Math.PI * 2); ctx.fill();
  }

  function paintRadial(cx, cy, radius) {
    ctx.globalCompositeOperation = erasing ? 'destination-out' : 'source-over';
    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(1, radius));
    grad.addColorStop(0, 'rgba(255,255,255,1)');
    grad.addColorStop(0.7, 'rgba(255,255,255,1)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.arc(cx, cy, Math.max(1, radius), 0, Math.PI * 2); ctx.fill();
  }

  function paintGradient(x0, y0, x1, y1) {
    ctx.globalCompositeOperation = erasing ? 'destination-out' : 'source-over';
    const grad = ctx.createLinearGradient(x0, y0, x1, y1);
    grad.addColorStop(0, 'rgba(255,255,255,1)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);
  }

  maskCanvas.addEventListener('pointerdown', (e) => {
    drawing = true;
    maskCanvas.setPointerCapture(e.pointerId);
    document.getElementById('paint-root').focus({preventScroll: true});
    [startX, startY] = toCanvasXY(e);
    lastX = startX; lastY = startY;
    lastShape = null;
    beforeShape = tool === 'brush' ? null : ctx.getImageData(0, 0, W, H);
    if (tool === 'brush') paintDab(startX, startY);
    e.preventDefault();
  });
  maskCanvas.addEventListener('pointermove', (e) => {
    if (!drawing) return;
    const [x, y] = toCanvasXY(e);
    if (tool === 'brush') {
      paintDab(x, y);
      lastX = x; lastY = y;
    } else {
      pctx.clearRect(0, 0, W, H);
      pctx.strokeStyle = 'rgba(255,255,255,0.9)'; pctx.lineWidth = 1.5; pctx.setLineDash([4, 3]);
      if (tool === 'radial') {
        const radius = Math.hypot(x - startX, y - startY);
        pctx.beginPath(); pctx.arc(startX, startY, radius, 0, Math.PI * 2); pctx.stroke();
      } else {
        pctx.beginPath(); pctx.moveTo(startX, startY); pctx.lineTo(x, y); pctx.stroke();
      }
    }
  });
  function endStroke(e) {
    if (!drawing) return;
    drawing = false;
    pctx.clearRect(0, 0, W, H);
    if (tool === 'radial') {
      const [x, y] = toCanvasXY(e);
      paintRadial(startX, startY, Math.hypot(x - startX, y - startY));
      lastShape = {tool: 'radial', x0: startX, y0: startY, x1: x, y1: y, erasing: erasing, before: beforeShape};
    } else if (tool === 'gradient') {
      const [x, y] = toCanvasXY(e);
      paintGradient(startX, startY, x, y);
      lastShape = {tool: 'gradient', x0: startX, y0: startY, x1: x, y1: y, erasing: erasing, before: beforeShape};
    }
    outlineShape();
  }
  maskCanvas.addEventListener('pointerup', endStroke);
  maskCanvas.addEventListener('pointerleave', endStroke);

  document.getElementById('commit-btn').addEventListener('click', () => {
    const base64 = maskCanvas.toDataURL('image/png').split(',')[1];
    commitToStreamlit(CONFIG.hiddenSelector, base64);
  });
})();
