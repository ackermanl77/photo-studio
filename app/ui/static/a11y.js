// Accessibility fixes for Streamlit's own markup (UI3), run in the page itself (app.ui.theme.
// inject_page_style()'s loader appends this once to the page's <head>) and kept applied by a
// MutationObserver, since Streamlit re-renders these nodes:
// - Material icon ligatures ("keyboard_arrow_right", "upload", "expand_more") are text spans that
//   screen readers read out: hidden from the accessibility tree.
// - Every slider was announced twice ("Exposure (EV) Exposure (EV)"): its input has an aria-label
//   and also aria-labelledby pointing at itself and its labelled group. The aria-label is enough.
// - The uploaders' hidden file input is named "file upload": it gets its uploader's label.
// - components.html() iframes have no name: the zero-size script ones (this loader, Find a tool's
//   scroll, the Library's keys) are hidden; an editor's gets its editor's own label (in place of
//   Streamlit's placeholder title "st.iframe").
(function() {
  if (window.__psA11yFixes) return;
  window.__psA11yFixes = true;
  function fix(root) {
    root.querySelectorAll('[data-testid="stIconMaterial"]:not([aria-hidden])').forEach((icon) => icon.setAttribute('aria-hidden', 'true'));
    root.querySelectorAll('[data-testid="stSlider"] input[type="range"][aria-labelledby][aria-label]').forEach((input) => input.removeAttribute('aria-labelledby'));
    root.querySelectorAll('[data-testid="stFileUploaderDropzoneInput"]').forEach((input) => {
      const zone = input.closest('[data-testid="stFileUploaderDropzone"]');
      const name = zone && zone.getAttribute('aria-label');
      if (name && input.getAttribute('aria-label') !== name) input.setAttribute('aria-label', name);
    });
    // Streamlit titles every one "st.iframe" -- no name to a screen reader.
    root.querySelectorAll('iframe:not([title]), iframe[title="st.iframe"]').forEach((frame) => {
      if (frame.getBoundingClientRect().height < 2) {
        frame.setAttribute('aria-hidden', 'true');
        frame.setAttribute('tabindex', '-1');
        return;
      }
      let label = null;
      try { label = frame.contentDocument && frame.contentDocument.querySelector('[aria-label]'); } catch (e) { /* not same-origin */ }
      if (label) frame.setAttribute('title', label.getAttribute('aria-label'));
    });
  }
  let pending = false;
  new MutationObserver(() => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; fix(document); });
  }).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['aria-labelledby', 'aria-label'] });
  fix(document);
})();
