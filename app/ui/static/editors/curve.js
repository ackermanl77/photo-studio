// Tone curve editor (app.ui_editors._curve_editor_html): five fixed points dragged vertically
// within their bounds, plus free extra points on the master curve; commits the whole curve as JSON.
(function() {
  const marginLeft = CONFIG.marginLeft, marginTop = CONFIG.marginTop, plotSize = CONFIG.plotSize, svgW = CONFIG.svgW;
  const bounds = CONFIG.bounds;
  const points = CONFIG.points;
  let extraPoints = CONFIG.extraPoints;
  const keys = CONFIG.keys, allowExtra = CONFIG.allowExtra;
  const svg = document.getElementById('curve-svg');
  const bg = document.getElementById('curve-bg');
  const line = document.getElementById('curve-line');
  const extraGroup = document.getElementById('extra-points');
  function toPx(x, y) { return [marginLeft + (x / 255) * plotSize, marginTop + plotSize - (y / 255) * plotSize]; }
  function fromPy(py) { return Math.max(0, Math.min(255, ((marginTop + plotSize - py) / plotSize) * 255)); }
  function fromPx(px) { return Math.max(0, Math.min(255, ((px - marginLeft) / plotSize) * 255)); }
  // Two points can't share an X (apply_curves needs strictly increasing X, and the server
  // would drop one): an extra point placed or dragged onto a used X moves to the nearest
  // free one, so what you see is what gets applied.
  function freeX(x, ownIdx) {
    const used = new Set(points.map(p => p[0]).concat(extraPoints.filter((_, i) => i !== ownIdx).map(p => p[0])));
    for (let d = 0; d <= 255; d++) {
      if (x + d <= 255 && !used.has(x + d)) return x + d;
      if (x - d >= 0 && !used.has(x - d)) return x - d;
    }
    return x;
  }
  function redraw() {
    const all = points.concat(extraPoints).slice().sort((a, b) => a[0] - b[0]);
    line.setAttribute('points', all.map(p => toPx(p[0], p[1]).join(',')).join(' '));
    svg.querySelectorAll('.pt.fixed').forEach((c, i) => {
      const [px, py] = toPx(points[i][0], points[i][1]);
      c.setAttribute('cx', px); c.setAttribute('cy', py);
    });
    extraGroup.innerHTML = '';
    extraPoints.forEach((p, i) => {
      const [px, py] = toPx(p[0], p[1]);
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('class', 'pt extra');
      circle.setAttribute('data-idx', i);
      circle.setAttribute('cx', px); circle.setAttribute('cy', py); circle.setAttribute('r', 7);
      extraGroup.appendChild(circle);
    });
  }
  function commit() {
    const data = { curve_enabled: true };
    keys.forEach((key, i) => { data[key] = points[i][1]; });
    if (allowExtra) data.curve_extra_points = extraPoints;
    commitToStreamlit('.st-key-curve-hidden', JSON.stringify(data));
  }

  // Fixed points: Y-only drag within their own [min, max]. Their count never changes, so
  // listeners attach once, directly.
  let dragIdx = null;
  svg.querySelectorAll('.pt.fixed').forEach((circle, i) => {
    circle.addEventListener('pointerdown', (e) => {
      dragIdx = i;
      circle.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
  });
  svg.addEventListener('pointermove', (e) => {
    if (dragIdx === null) return;
    const rect = svg.getBoundingClientRect();
    const scale = svgW / rect.width;
    const py = (e.clientY - rect.top) * scale;
    let value = Math.round(fromPy(py));
    const [lo, hi] = bounds[dragIdx];
    value = Math.max(lo, Math.min(hi, value));
    points[dragIdx][1] = value;
    redraw();
  });
  function endDrag() {
    if (dragIdx === null) return;
    dragIdx = null;
    commit();
  }
  svg.addEventListener('pointerup', endDrag);
  svg.addEventListener('pointerleave', endDrag);

  // Keyboard (F11): a focused fixed point moves with Up/Down (Shift = 10 levels) within its
  // own [min, max]; one commit per burst of presses.
  let keyTimer = null;
  svg.querySelectorAll('.pt.fixed').forEach((circle, i) => {
    circle.addEventListener('keydown', (e) => {
      const step = (e.key === 'ArrowUp' ? 1 : e.key === 'ArrowDown' ? -1 : 0) * (e.shiftKey ? 10 : 1);
      if (!step) return;
      e.preventDefault();
      const [lo, hi] = bounds[i];
      points[i][1] = Math.max(lo, Math.min(hi, points[i][1] + step));
      circle.setAttribute('aria-valuenow', points[i][1]);
      redraw();
      clearTimeout(keyTimer);
      keyTimer = setTimeout(commit, CONFIG.keyCommitDelay);
    });
  });

  // Extra points: added/removed dynamically, so handled via event delegation on `svg`
  // (checking e.target) instead of per-circle listeners that would need re-attaching every
  // time redraw() rebuilds the #extra-points group's children.
  let dragExtraIdx = null;
  svg.addEventListener('pointerdown', (e) => {
    const el = e.target.closest('.pt.extra');
    if (!el) return;
    dragExtraIdx = Number(el.dataset.idx);
    svg.setPointerCapture(e.pointerId);
    e.preventDefault();
    e.stopPropagation();
  });
  svg.addEventListener('pointermove', (e) => {
    if (dragExtraIdx === null) return;
    const rect = svg.getBoundingClientRect();
    const scale = svgW / rect.width;
    const px = (e.clientX - rect.left) * scale;
    const py = (e.clientY - rect.top) * scale;
    extraPoints[dragExtraIdx] = [freeX(Math.round(fromPx(px)), dragExtraIdx), Math.round(fromPy(py))];
    redraw();
  });
  function endExtraDrag() {
    if (dragExtraIdx === null) return;
    dragExtraIdx = null;
    commit();
  }
  svg.addEventListener('pointerup', endExtraDrag);
  svg.addEventListener('pointerleave', endExtraDrag);
  svg.addEventListener('dblclick', (e) => {
    const el = e.target.closest('.pt.extra');
    if (!el) return;
    extraPoints.splice(Number(el.dataset.idx), 1);
    redraw();
    commit();
  });

  // Click empty background (not a drag-release from a point, which targets a circle
  // element instead of the rect) adds a new extra point at the clicked position.
  bg.addEventListener('click', (e) => {
    if (!allowExtra) return;
    const rect = svg.getBoundingClientRect();
    const scale = svgW / rect.width;
    const px = (e.clientX - rect.left) * scale;
    const py = (e.clientY - rect.top) * scale;
    extraPoints.push([freeX(Math.round(fromPx(px)), -1), Math.round(fromPy(py))]);
    redraw();
    commit();
  });

  redraw();
})();
