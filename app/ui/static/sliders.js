// Slider fills that start at zero (UI4), run in the page itself like a11y.js. Streamlit fills every
// slider from its left end, so a -100..+100 control at 0 looked half "on". A slider whose range
// spans zero gets its fill from zero to the value instead, in Streamlit's own two track colours
// (read once from its gradient), and a tick at zero. Tracks with a meaning of their own (temperature,
// hue...: app.ui.theme.slider_tracks_style(), marked --ps-track) are left as they are.
(function() {
  if (window.__psSliderFills) return;
  window.__psSliderFills = true;
  let neutral = null, accent = null;
  function fill(root) {
    root.querySelectorAll('[data-testid="stSlider"]').forEach((slider) => {
      const input = slider.querySelector('input[type="range"]');
      const track = slider.querySelector('[role="group"] > div > div:first-child');
      if (!input || !track || getComputedStyle(track).getPropertyValue('--ps-track').trim()) return;
      const min = Number(input.min), max = Number(input.max), value = Number(input.value);
      if (!(min < 0 && max > 0) || Number.isNaN(value)) return;
      if (!accent) {
        const stops = getComputedStyle(track).backgroundImage.match(/rgba?\([^)]*\)/g) || [];
        if (stops.length < 2) return;
        [neutral, accent] = stops;
      }
      const zero = (-min / (max - min)) * 100, at = ((value - min) / (max - min)) * 100;
      const from = Math.min(zero, at).toFixed(2), to = Math.max(zero, at).toFixed(2);
      const image = `linear-gradient(to right, ${neutral} ${from}%, ${accent} ${from}%, ${accent} ${to}%, ${neutral} ${to}%)`;
      if (track.style.getPropertyValue('background-image') !== image) track.style.setProperty('background-image', image, 'important');
      track.classList.add('ps-zero-track');
      // Only real changes: the observer watches style attributes (a drag moves the thumb's).
      if (track.style.getPropertyValue('--ps-zero') !== zero.toFixed(2) + '%') track.style.setProperty('--ps-zero', zero.toFixed(2) + '%');
    });
  }
  let pending = false;
  new MutationObserver(() => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; fill(document); });
  }).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['value', 'aria-valuetext', 'style'] });
  fill(document);
})();
