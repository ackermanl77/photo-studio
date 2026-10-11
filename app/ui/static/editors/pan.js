// Zoomed canvas with drag-to-pan (app.ui_editors._pan_viewer_html): on release (or after a burst of
// arrow keys) the view's new center is committed as {x, y} fractions of the full image.
(function() {
  const wrap = document.getElementById('pan-wrap');
  const img = document.getElementById('pan-img');
  const VIEW_W = CONFIG.viewW, VIEW_H = CONFIG.viewH, ZOOM = CONFIG.zoom, IMG_W = CONFIG.imageW, IMG_H = CONFIG.imageH;
  const center = {x: CONFIG.center[0], y: CONFIG.center[1]};
  let start = null;

  function commit(x, y) {
    commitToStreamlit(CONFIG.hiddenSelector, JSON.stringify({x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y))}));
    // If the rerun leaves this exact view (already at an edge), don't stay visually offset.
    setTimeout(() => { img.style.transform = ''; }, 1500);
  }

  // Screen pixels moved -> fractions of the full image (one view pixel = 1/ZOOM image pixels).
  function shifted(dx, dy) {
    const rect = img.getBoundingClientRect();
    const imageDx = dx * (VIEW_W / Math.max(1, rect.width)) / ZOOM;
    const imageDy = dy * (VIEW_H / Math.max(1, rect.height)) / ZOOM;
    return [center.x - imageDx / IMG_W, center.y - imageDy / IMG_H];
  }

  wrap.addEventListener('pointerdown', (e) => {
    start = {x: e.clientX, y: e.clientY};
    wrap.setPointerCapture(e.pointerId);
    wrap.style.cursor = 'grabbing';
    wrap.focus({preventScroll: true});
    e.preventDefault();
  });
  wrap.addEventListener('pointermove', (e) => {
    if (!start) return;
    img.style.transform = 'translate(' + (e.clientX - start.x) + 'px,' + (e.clientY - start.y) + 'px)';
  });
  function endDrag(e) {
    if (!start) return;
    const dx = e.clientX - start.x, dy = e.clientY - start.y;
    start = null;
    wrap.style.cursor = 'grab';
    if (Math.abs(dx) + Math.abs(dy) < 3) { img.style.transform = ''; return; }
    const [x, y] = shifted(dx, dy);
    commit(x, y);
  }
  wrap.addEventListener('pointerup', endDrag);
  wrap.addEventListener('pointercancel', endDrag);

  const ARROWS = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]};
  let keyTimer = null, pending = [0, 0];
  wrap.addEventListener('keydown', (e) => {
    const move = ARROWS[e.key];
    if (!move) return;
    e.preventDefault();
    const rect = img.getBoundingClientRect();
    const fraction = e.shiftKey ? 0.5 : 0.1;
    // Arrow = look further that way, i.e. the photo moves the opposite way.
    pending[0] -= move[0] * rect.width * fraction;
    pending[1] -= move[1] * rect.height * fraction;
    img.style.transform = 'translate(' + pending[0] + 'px,' + pending[1] + 'px)';
    clearTimeout(keyTimer);
    keyTimer = setTimeout(() => {
      const [x, y] = shifted(pending[0], pending[1]);
      pending = [0, 0];
      commit(x, y);
    }, CONFIG.keyCommitDelay);
  });
})();
