// Free-form crop box (app.ui_editors._crop_editor_html): two corner handles plus drag-to-move; the
// box is kept as fractions of the photo and committed as {x0, y0, x1, y1}.
(function() {
  const wrap = document.getElementById('crop-wrap');
  const rectEl = document.getElementById('crop-rect');
  const tl = document.getElementById('crop-tl');
  const br = document.getElementById('crop-br');
  const dimsLabel = document.getElementById('crop-dims');
  const box = Object.assign({}, CONFIG.box);
  const snapX = CONFIG.snapX;
  const snapY = CONFIG.snapY;
  const fullW = CONFIG.fullW, fullH = CONFIG.fullH;
  const SNAP_THRESHOLD = 0.012;
  let dragMode = null, moveStart = null;

  function snapValue(value, candidates) {
    for (const candidate of candidates) {
      if (Math.abs(value - candidate) < SNAP_THRESHOLD) return candidate;
    }
    return value;
  }

  function redraw() {
    rectEl.style.left = (box.x0 * 100) + '%';
    rectEl.style.top = (box.y0 * 100) + '%';
    rectEl.style.width = ((box.x1 - box.x0) * 100) + '%';
    rectEl.style.height = ((box.y1 - box.y0) * 100) + '%';
    tl.style.left = (box.x0 * 100) + '%';
    tl.style.top = (box.y0 * 100) + '%';
    br.style.left = (box.x1 * 100) + '%';
    br.style.top = (box.y1 * 100) + '%';
    const pxW = Math.round((box.x1 - box.x0) * fullW);
    const pxH = Math.round((box.y1 - box.y0) * fullH);
    dimsLabel.textContent = CONFIG.text.crop + ' ' + pxW + '×' + pxH + 'px -- ' + CONFIG.text.hint;
  }
  redraw();

  function fractionalPos(e) {
    const rect = wrap.getBoundingClientRect();
    const fx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const fy = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    return [fx, fy];
  }

  function startDrag(mode) {
    return function(e) {
      dragMode = mode;
      e.currentTarget.focus({preventScroll: true});  // so the arrow keys then nudge what was dragged
      wrap.setPointerCapture(e.pointerId);
      if (mode === 'move') {
        const [fx, fy] = fractionalPos(e);
        moveStart = {fx: fx, fy: fy, x0: box.x0, y0: box.y0, x1: box.x1, y1: box.y1};
      }
      e.preventDefault();
      e.stopPropagation();
    };
  }
  tl.addEventListener('pointerdown', startDrag('tl'));
  br.addEventListener('pointerdown', startDrag('br'));
  rectEl.addEventListener('pointerdown', startDrag('move'));

  const MIN_SIZE = 0.02;
  wrap.addEventListener('pointermove', function(e) {
    if (!dragMode) return;
    let [fx, fy] = fractionalPos(e);
    if (dragMode === 'tl' || dragMode === 'br') {
      // Snap only the corner handles -- not a body drag/move (a documented scope cut).
      fx = snapValue(fx, snapX);
      fy = snapValue(fy, snapY);
    }
    if (dragMode === 'tl') {
      box.x0 = Math.min(fx, box.x1 - MIN_SIZE);
      box.y0 = Math.min(fy, box.y1 - MIN_SIZE);
    } else if (dragMode === 'br') {
      box.x1 = Math.max(fx, box.x0 + MIN_SIZE);
      box.y1 = Math.max(fy, box.y0 + MIN_SIZE);
    } else if (dragMode === 'move') {
      const w = moveStart.x1 - moveStart.x0, h = moveStart.y1 - moveStart.y0;
      let nx0 = Math.max(0, Math.min(1 - w, moveStart.x0 + (fx - moveStart.fx)));
      let ny0 = Math.max(0, Math.min(1 - h, moveStart.y0 + (fy - moveStart.fy)));
      box.x0 = nx0; box.y0 = ny0; box.x1 = nx0 + w; box.y1 = ny0 + h;
    }
    redraw();
  });

  function commit() {
    commitToStreamlit(CONFIG.hiddenSelector, JSON.stringify(box));
  }
  function endDrag() {
    if (!dragMode) return;
    dragMode = null;
    commit();
  }
  wrap.addEventListener('pointerup', endDrag);
  wrap.addEventListener('pointerleave', endDrag);

  // Keyboard (F11): the focused corner or the whole box moves by one source pixel per arrow
  // press (Shift = 10), with the same limits as a drag; one commit per burst of presses.
  const STEP_X = 1 / Math.max(1, fullW), STEP_Y = 1 / Math.max(1, fullH);
  const ARROWS = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]};
  let keyTimer = null;
  function nudge(target, dx, dy) {
    if (target === 'tl') {
      box.x0 = Math.max(0, Math.min(box.x1 - MIN_SIZE, box.x0 + dx));
      box.y0 = Math.max(0, Math.min(box.y1 - MIN_SIZE, box.y0 + dy));
    } else if (target === 'br') {
      box.x1 = Math.min(1, Math.max(box.x0 + MIN_SIZE, box.x1 + dx));
      box.y1 = Math.min(1, Math.max(box.y0 + MIN_SIZE, box.y1 + dy));
    } else {
      const w = box.x1 - box.x0, h = box.y1 - box.y0;
      box.x0 = Math.max(0, Math.min(1 - w, box.x0 + dx));
      box.y0 = Math.max(0, Math.min(1 - h, box.y0 + dy));
      box.x1 = box.x0 + w; box.y1 = box.y0 + h;
    }
  }
  wrap.addEventListener('keydown', (e) => {
    const move = ARROWS[e.key];
    if (!move) return;
    e.preventDefault();
    const target = e.target === tl ? 'tl' : e.target === br ? 'br' : 'move';
    const factor = e.shiftKey ? 10 : 1;
    nudge(target, move[0] * STEP_X * factor, move[1] * STEP_Y * factor);
    redraw();
    clearTimeout(keyTimer);
    keyTimer = setTimeout(commit, CONFIG.keyCommitDelay);
  });
})();
