// Pen tool (app.ui_editors._pen_editor_html): click to add anchors, drag them or their (symmetric)
// curve handle, double-click or Delete to remove; commits the point list.
(function() {
  const wrap = document.getElementById('pen-wrap');
  const bg = document.getElementById('pen-bg');
  const pathEl = document.getElementById('pen-path');
  const handleLines = document.getElementById('pen-handle-lines');
  const anchorLayer = document.getElementById('pen-anchors');
  const handleLayer = document.getElementById('pen-handles');
  let points = CONFIG.points;
  const closed = CONFIG.closed;
  let dragMode = null;  // {type: 'anchor'|'handle', idx: number}
  let selectedIdx = null;  // the anchor the arrow keys move (F11)

  function fractionalPos(e) {
    const rect = wrap.getBoundingClientRect();
    const fx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const fy = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    return [fx * 100, fy * 100];
  }

  // Mirrors app.vector_shapes.path_outline_points()/_cubic_bezier_points() exactly -- the
  // preview drawn while editing and the layer actually rendered by the backend must trace
  // the same curve, or a point could look right here and land somewhere else once applied.
  function cubicPoints(p0, c1, c2, p1, steps) {
    const pts = [];
    for (let i = 0; i <= steps; i++) {
      const t = i / steps, mt = 1 - t;
      const x = mt*mt*mt*p0[0] + 3*mt*mt*t*c1[0] + 3*mt*t*t*c2[0] + t*t*t*p1[0];
      const y = mt*mt*mt*p0[1] + 3*mt*mt*t*c1[1] + 3*mt*t*t*c2[1] + t*t*t*p1[1];
      pts.push([x, y]);
    }
    return pts;
  }

  function outline() {
    if (points.length < 2) return points.map(p => [p.x_pct, p.y_pct]);
    const anchors = points.map(p => [p.x_pct, p.y_pct]);
    const handles = points.map(p => [p.handle_dx_pct || 0, p.handle_dy_pct || 0]);
    const n = anchors.length;
    const segCount = closed ? n : n - 1;
    let out = [anchors[0]];
    for (let i = 0; i < segCount; i++) {
      const p0 = anchors[i], p1 = anchors[(i + 1) % n];
      const h0 = handles[i], h1 = handles[(i + 1) % n];
      const c1 = [p0[0] + h0[0], p0[1] + h0[1]];
      const c2 = [p1[0] - h1[0], p1[1] - h1[1]];
      out = out.concat(cubicPoints(p0, c1, c2, p1, 20).slice(1));
    }
    return out;
  }

  function redraw() {
    const out = outline();
    const d = out.length ? ('M ' + out.map(p => p[0].toFixed(2) + ' ' + p[1].toFixed(2)).join(' L ')) + (closed ? ' Z' : '') : '';
    pathEl.setAttribute('d', d);
    pathEl.setAttribute('fill', closed ? 'rgba(96,165,250,0.25)' : 'none');

    // Rebuilding the dots drops keyboard focus, so remember whether an anchor had it.
    const hadFocus = document.activeElement && document.activeElement.classList.contains('pen-anchor');
    anchorLayer.innerHTML = '';
    handleLayer.innerHTML = '';
    handleLines.innerHTML = '';
    points.forEach((p, i) => {
      const anchor = document.createElement('div');
      anchor.className = 'pen-anchor' + (i === selectedIdx ? ' selected' : '');
      anchor.style.left = p.x_pct + '%';
      anchor.style.top = p.y_pct + '%';
      anchor.dataset.idx = i;
      anchor.tabIndex = 0;
      anchor.setAttribute('aria-label', CONFIG.text.point.replace('{0}', i + 1).replace('{1}', points.length));
      anchor.addEventListener('focus', () => { selectedIdx = i; });
      anchorLayer.appendChild(anchor);
      if (hadFocus && i === selectedIdx) anchor.focus({preventScroll: true});

      const hx = p.x_pct + (p.handle_dx_pct || 0);
      const hy = p.y_pct + (p.handle_dy_pct || 0);
      const handle = document.createElement('div');
      handle.className = 'pen-handle';
      handle.style.left = hx + '%';
      handle.style.top = hy + '%';
      handle.dataset.idx = i;
      handleLayer.appendChild(handle);

      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', p.x_pct); line.setAttribute('y1', p.y_pct);
      line.setAttribute('x2', hx); line.setAttribute('y2', hy);
      handleLines.appendChild(line);
    });
  }
  redraw();

  function commit() {
    commitToStreamlit(CONFIG.hiddenSelector, JSON.stringify(points));
  }

  anchorLayer.addEventListener('pointerdown', (e) => {
    const el = e.target.closest('.pen-anchor');
    if (!el) return;
    dragMode = {type: 'anchor', idx: Number(el.dataset.idx)};
    selectedIdx = dragMode.idx;
    wrap.focus({preventScroll: true});  // keyboard nudging then applies to this point
    wrap.setPointerCapture(e.pointerId);
    e.preventDefault();
    e.stopPropagation();
  });
  handleLayer.addEventListener('pointerdown', (e) => {
    const el = e.target.closest('.pen-handle');
    if (!el) return;
    dragMode = {type: 'handle', idx: Number(el.dataset.idx)};
    wrap.setPointerCapture(e.pointerId);
    e.preventDefault();
    e.stopPropagation();
  });
  wrap.addEventListener('pointermove', (e) => {
    if (!dragMode) return;
    const [fx, fy] = fractionalPos(e);
    const p = points[dragMode.idx];
    if (dragMode.type === 'anchor') {
      p.x_pct = fx; p.y_pct = fy;
    } else {
      p.handle_dx_pct = fx - p.x_pct;
      p.handle_dy_pct = fy - p.y_pct;
    }
    redraw();
  });
  function endDrag() {
    if (!dragMode) return;
    dragMode = null;
    commit();
  }
  wrap.addEventListener('pointerup', endDrag);
  wrap.addEventListener('pointerleave', endDrag);

  anchorLayer.addEventListener('dblclick', (e) => {
    const el = e.target.closest('.pen-anchor');
    if (!el) return;
    points.splice(Number(el.dataset.idx), 1);
    redraw();
    commit();
  });

  // Attached to the background image itself, not `wrap`: anchor/handle dots are siblings layered
  // on top of it, so a click on one of them never bubbles into "add a point here".
  bg.addEventListener('click', (e) => {
    const [fx, fy] = fractionalPos(e);
    points.push({x_pct: fx, y_pct: fy, handle_dx_pct: 0, handle_dy_pct: 0});
    selectedIdx = points.length - 1;
    wrap.focus({preventScroll: true});
    redraw();
    commit();
  });

  // Keyboard (F11): arrows nudge the selected anchor by one display pixel (Shift = 10),
  // Alt+arrows its curve handle, Delete/Backspace removes it; one commit per burst.
  const ARROWS = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]};
  let keyTimer = null;
  function scheduleCommit() {
    clearTimeout(keyTimer);
    keyTimer = setTimeout(commit, CONFIG.keyCommitDelay);
  }
  wrap.addEventListener('keydown', (e) => {
    if (selectedIdx === null || !points[selectedIdx]) return;
    const p = points[selectedIdx];
    if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault();
      points.splice(selectedIdx, 1);
      selectedIdx = points.length ? Math.min(selectedIdx, points.length - 1) : null;
      redraw();
      scheduleCommit();
      return;
    }
    const move = ARROWS[e.key];
    if (!move) return;
    e.preventDefault();
    const rect = wrap.getBoundingClientRect();
    const stepX = 100 / Math.max(1, rect.width) * (e.shiftKey ? 10 : 1);
    const stepY = 100 / Math.max(1, rect.height) * (e.shiftKey ? 10 : 1);
    if (e.altKey) {
      p.handle_dx_pct = (p.handle_dx_pct || 0) + move[0] * stepX;
      p.handle_dy_pct = (p.handle_dy_pct || 0) + move[1] * stepY;
    } else {
      p.x_pct = Math.max(0, Math.min(100, p.x_pct + move[0] * stepX));
      p.y_pct = Math.max(0, Math.min(100, p.y_pct + move[1] * stepY));
    }
    redraw();
    scheduleCommit();
  });
})();
