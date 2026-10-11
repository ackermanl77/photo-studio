// Before/after slider: one range input drives a clip-path on the "before" image (no commit).
(function() {
  const range = document.getElementById('ba-range');
  const before = document.getElementById('ba-before');
  const handle = document.getElementById('ba-handle');
  range.addEventListener('input', () => {
    const v = range.value;
    before.style.clipPath = `inset(0 ${100 - v}% 0 0)`;
    handle.style.left = v + '%';
  });
})();
