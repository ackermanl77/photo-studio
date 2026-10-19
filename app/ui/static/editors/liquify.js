// Liquify brush (app.ui_editors._liquify_editor_html): drag on the photo to record one stroke --
// its points as fractions of the frame -- and commit {points} when the pointer lifts; Python adds
// the tool, size and strength chosen beside the editor and the backend warps the photo (app.liquify).
// The canvas only draws the brush outline and the stroke being recorded.
// Keyboard (F11): arrow keys move a brush cursor (Shift = 10x), Space or Enter applies one stamp there.
(function() {
  const wrap = document.getElementById('liq-wrap');
  const canvas = document.getElementById('liq-canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const MAX_POINTS = CONFIG.maxPoints;
  let points = null, cursor = {x: 0.5, y: 0.5};

  function radiusPx() { return CONFIG.size * Math.min(W, H); }
  function toFraction(e) {
    const rect = canvas.getBoundingClientRect();
    return {x: Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)), y: Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height))};
  }
  function draw() {
    ctx.clearRect(0, 0, W, H);
    ctx.lineWidth = 1.5;
    if (points && points.length > 1) {
      ctx.strokeStyle = 'rgba(56,189,248,0.9)';
      ctx.beginPath();
      ctx.moveTo(points[0][0] * W, points[0][1] * H);
      points.slice(1).forEach((p) => ctx.lineTo(p[0] * W, p[1] * H));
      ctx.stroke();
    }
    ctx.strokeStyle = 'rgba(251,191,36,0.95)';
    ctx.beginPath();
    ctx.arc(cursor.x * W, cursor.y * H, radiusPx(), 0, Math.PI * 2);
    ctx.stroke();
  }
  function thinned(list) {
    if (list.length <= MAX_POINTS) return list;
    const out = [];
    for (let i = 0; i < MAX_POINTS - 1; i++) out.push(list[Math.floor(i * (list.length - 1) / (MAX_POINTS - 1))]);
    out.push(list[list.length - 1]);
    return out;
  }
  function commit(list) {
    commitToStreamlit(CONFIG.hiddenSelector, JSON.stringify({points: thinned(list).map((p) => [Number(p[0].toFixed(5)), Number(p[1].toFixed(5))])}));
  }

  canvas.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    if (canvas.setPointerCapture) canvas.setPointerCapture(e.pointerId);
    cursor = toFraction(e);
    points = [[cursor.x, cursor.y]];
    draw();
  });
  canvas.addEventListener('pointermove', (e) => {
    cursor = toFraction(e);
    if (points) {
      const last = points[points.length - 1];
      // Record a point every ~2 display pixels; the backend spaces its stamps itself.
      if (Math.hypot((cursor.x - last[0]) * W, (cursor.y - last[1]) * H) >= 2) points.push([cursor.x, cursor.y]);
    }
    draw();
  });
  function finish() {
    if (!points) return;
    const list = points;
    points = null;
    draw();
    commit(list);
  }
  canvas.addEventListener('pointerup', finish);
  canvas.addEventListener('pointercancel', () => { points = null; draw(); });

  wrap.addEventListener('keydown', (e) => {
    const step = (e.shiftKey ? 10 : 1);
    const moves = {ArrowLeft: [-step / W, 0], ArrowRight: [step / W, 0], ArrowUp: [0, -step / H], ArrowDown: [0, step / H]};
    if (moves[e.key]) {
      e.preventDefault();
      cursor = {x: Math.max(0, Math.min(1, cursor.x + moves[e.key][0])), y: Math.max(0, Math.min(1, cursor.y + moves[e.key][1]))};
      draw();
    } else if (e.key === ' ' || e.key === 'Enter') {
      e.preventDefault();
      commit([[cursor.x, cursor.y]]);
    }
  });
  draw();
})();
