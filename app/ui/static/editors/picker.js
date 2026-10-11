// Click picker (Magic Wand, white balance, zoom navigator -- app.ui_editors._wand_picker_html): one
// click commits its fractional {x, y} position on the photo.
(function() {
  const img = document.getElementById('wand-img');
  const marker = document.getElementById('wand-marker');
  img.addEventListener('click', function(e) {
    const rect = img.getBoundingClientRect();
    const fx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const fy = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    marker.style.left = (fx * rect.width) + 'px';
    marker.style.top = (fy * rect.height) + 'px';
    marker.style.display = 'block';
    commitToStreamlit(CONFIG.hiddenSelector, JSON.stringify({x: fx, y: fy}));
  });
})();
