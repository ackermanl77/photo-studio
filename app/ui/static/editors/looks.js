// Look gallery (app.ui_editors._look_gallery_html, UI5): the built-in looks and the saved presets as
// cards, in a grid or a list (remembered in this browser). Hovering or focusing a card shows its
// preview on the canvas itself -- the page's photo is swapped for the preview until the pointer
// leaves, nothing is rendered or committed; clicking (or Enter) applies it through the hidden field,
// so the app records it as one history step.
(function() {
  const groupsEl = document.getElementById('looks-groups');
  let view = 'grid';
  try { view = window.localStorage.getItem('ps-looks-view') || 'grid'; } catch (e) { /* storage blocked */ }

  function canvasImage() {
    try { return window.parent.document.querySelector('.st-key-canvas-stage img'); } catch (e) { return null; }
  }
  function badge(stage, text) {
    let el = stage.querySelector('.ps-look-badge');
    if (!el && text) {
      el = window.parent.document.createElement('div');
      el.className = 'ps-look-badge';
      stage.appendChild(el);
    }
    if (el && text) el.textContent = text;
    if (el && !text) el.remove();
  }
  function preview(card) {
    const img = canvasImage();
    if (!img) return;
    if (!img.dataset.psOriginal) img.dataset.psOriginal = img.src;
    img.src = 'data:image/jpeg;base64,' + card.preview;
    badge(img.closest('.st-key-canvas-stage'), CONFIG.previewLabel.replace('{0}', card.title));
  }
  function restore() {
    const img = canvasImage();
    if (!img || !img.dataset.psOriginal) return;
    img.src = img.dataset.psOriginal;
    delete img.dataset.psOriginal;
    badge(img.closest('.st-key-canvas-stage'), null);
  }

  function render() {
    groupsEl.innerHTML = '';
    document.querySelectorAll('.view-btn').forEach((btn) => btn.classList.toggle('active', btn.dataset.view === view));
    CONFIG.groups.forEach((group) => {
      const title = document.createElement('div');
      title.className = 'group-title';
      title.textContent = group.title;
      groupsEl.appendChild(title);
      const cards = document.createElement('div');
      cards.className = 'cards ' + view;
      group.cards.forEach((card) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'card' + (card.current ? ' current' : '');
        button.setAttribute('aria-label', card.title);
        const img = document.createElement('img');
        img.src = 'data:image/jpeg;base64,' + card.thumb;
        img.alt = '';
        const name = document.createElement('span');
        name.textContent = card.title;
        button.appendChild(img);
        button.appendChild(name);
        button.addEventListener('mouseenter', () => preview(card));
        button.addEventListener('focus', () => preview(card));
        button.addEventListener('mouseleave', restore);
        button.addEventListener('blur', restore);
        button.addEventListener('click', () => {
          restore();
          // A time stamp, so applying the same card again after other edits is a new commit.
          commitToStreamlit(CONFIG.hiddenSelector, JSON.stringify({kind: card.kind, name: card.name, at: Date.now()}));
        });
        cards.appendChild(button);
      });
      groupsEl.appendChild(cards);
    });
  }
  document.querySelectorAll('.view-btn').forEach((btn) => btn.addEventListener('click', () => {
    view = btn.dataset.view;
    try { window.localStorage.setItem('ps-looks-view', view); } catch (e) { /* storage blocked */ }
    render();
  }));
  window.addEventListener('pagehide', restore);
  render();
})();
