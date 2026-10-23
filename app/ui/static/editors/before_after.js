// Before/after split (Photoshop's Quick Compare): the "before" image is clipped to one side of the
// divider. Side by side (↔) or top and bottom (↕), remembered in this browser; drag on the photo, or
// use the range input (keyboard), to move the divider. View-only: nothing is committed.
(function() {
  const root = document.getElementById('ba-root');
  const range = document.getElementById('ba-range');
  const before = document.getElementById('ba-before');
  const handle = document.getElementById('ba-handle');
  const dirBtn = document.getElementById('ba-dir');
  let vertical = false;
  try { vertical = window.localStorage.getItem('ps-ba-dir') === 'v'; } catch (e) { /* storage blocked */ }

  function show() {
    const v = Number(range.value);
    root.classList.toggle('ba-v', vertical);
    root.classList.toggle('ba-h', !vertical);
    dirBtn.textContent = vertical ? '↔' : '↕';
    before.style.clipPath = vertical ? `inset(0 0 ${100 - v}% 0)` : `inset(0 ${100 - v}% 0 0)`;
    handle.style.left = vertical ? '' : v + '%';
    handle.style.top = vertical ? v + '%' : '';
  }
  range.addEventListener('input', show);
  dirBtn.addEventListener('click', () => {
    vertical = !vertical;
    try { window.localStorage.setItem('ps-ba-dir', vertical ? 'v' : 'h'); } catch (e) { /* storage blocked */ }
    show();
  });

  let dragging = false;
  function moveTo(e) {
    const rect = root.getBoundingClientRect();
    const fraction = vertical ? (e.clientY - rect.top) / rect.height : (e.clientX - rect.left) / rect.width;
    range.value = Math.round(Math.max(0, Math.min(1, fraction)) * 100);
    show();
  }
  root.addEventListener('pointerdown', (e) => {
    if (e.target === dirBtn || e.target === range) return;
    dragging = true;
    root.setPointerCapture(e.pointerId);
    moveTo(e);
  });
  root.addEventListener('pointermove', (e) => { if (dragging) moveTo(e); });
  root.addEventListener('pointerup', () => { dragging = false; });
  show();
})();
