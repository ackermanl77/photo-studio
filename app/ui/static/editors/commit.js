// Shared by every on-canvas editor (app/ui_editors.py): commit `payload` into the hidden
// st.text_area inside the parent page's container matching `selector`, then press Ctrl+Enter on it.
// Confirmed live: st.text_area only commits -- and reruns the script -- on Ctrl+Enter or on losing
// focus, and a synthetic .blur() alone wasn't reliable enough in this Streamlit version, so the
// documented Ctrl+Enter shortcut is dispatched directly. A script-dispatched event on a slider does
// NOT reliably reach the server, which is why no editor touches a native widget (see CLAUDE.md).
function commitToStreamlit(selector, payload) {
  try {
    const container = window.parent.document.querySelector(selector);
    if (!container) return false;
    const textarea = container.querySelector('textarea');
    if (!textarea) return false;
    const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
    textarea.focus();
    setter.call(textarea, payload);
    textarea.dispatchEvent(new Event('input', { bubbles: true }));
    textarea.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', ctrlKey: true, bubbles: true, cancelable: true }));
    textarea.dispatchEvent(new KeyboardEvent('keyup', { key: 'Enter', code: 'Enter', ctrlKey: true, bubbles: true, cancelable: true }));
    textarea.blur();
    return true;
  } catch (e) {
    return false;
  }
}
