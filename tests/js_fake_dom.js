// A minimal fake DOM, just enough to run the on-canvas editors' <script> (app/ui_editors.py) under
// Node in tests/test_interaction_improvements.py: elements with listeners/style/classList, a fixed
// 400x300 layout, a no-op 2D canvas, a fake setTimeout queue (flushTimers()) and the parent
// page's hidden text_area, whose committed values are recorded in `commits`.
'use strict';
const commits = [];
const canvasCalls = [];
const timers = [];
global.setTimeout = (fn) => { timers.push(fn); return timers.length; };
global.clearTimeout = (id) => { if (id) timers[id - 1] = null; };
function flushTimers() { while (timers.some(Boolean)) { const pending = timers.splice(0); pending.forEach((fn) => fn && fn()); } }

class FakeElement {
  constructor(tag, id) {
    this.tagName = (tag || 'div').toUpperCase(); this.id = id || ''; this.children = []; this.parent = null;
    this.style = {}; this.dataset = {}; this.attributes = {}; this.listeners = {}; this.tabIndex = -1;
    this.className = ''; this.width = 400; this.height = 300; this.value = '';
    this.min = '4'; this.max = '80';
    const self = this;
    this.classList = {
      contains: (c) => self.className.split(/\s+/).includes(c),
      add: (c) => { if (!self.classList.contains(c)) self.className = (self.className + ' ' + c).trim(); },
      remove: (c) => { self.className = self.className.split(/\s+/).filter((x) => x !== c).join(' '); },
      toggle: (c, on) => { (on === undefined ? !self.classList.contains(c) : on) ? self.classList.add(c) : self.classList.remove(c); },
    };
  }
  set innerHTML(_v) { this.children = []; }
  get innerHTML() { return ''; }
  setAttribute(k, v) { this.attributes[k] = String(v); }
  getAttribute(k) { return this.attributes[k]; }
  appendChild(child) { child.parent = this; this.children.push(child); return child; }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  dispatch(type, props) {
    const event = Object.assign({ type, target: this, currentTarget: this, defaultPrevented: false, pointerId: 1,
      preventDefault() { this.defaultPrevented = true; }, stopPropagation() { this.stopped = true; } }, props || {});
    let reachedDocument = false;
    for (let node = this; node && !event.stopped; node = node.parent) {
      event.currentTarget = node;
      reachedDocument = reachedDocument || node === document;
      (node.listeners[type] || []).forEach((fn) => fn(event));
    }
    if (!event.stopped && !reachedDocument) { event.currentTarget = document; (document.listeners[type] || []).forEach((fn) => fn(event)); }
    return event;
  }
  matches(selector) {
    return selector.split(',').some((part) => {
      const classes = (part.trim().match(/\.[\w-]+/g) || []).map((c) => c.slice(1));
      return classes.length > 0 && classes.every((c) => this.classList.contains(c));
    });
  }
  closest(selector) { for (let node = this; node; node = node.parent) if (node.matches && node.matches(selector)) return node; return null; }
  querySelectorAll(selector) {
    const out = [];
    const walk = (node) => node.children.forEach((child) => { if (child.matches(selector)) out.push(child); walk(child); });
    walk(this);
    return out;
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  getBoundingClientRect() { return { left: 0, top: 0, width: 400, height: 300 }; }
  setPointerCapture() {}
  focus() { document.activeElement = this; (this.listeners.focus || []).forEach((fn) => fn({ target: this })); }
  blur() {}
  getContext() {
    const gradient = { addColorStop() {} };
    return new Proxy({}, { get: (_t, name) => {
      if (name === 'createRadialGradient' || name === 'createLinearGradient') return () => gradient;
      if (name === 'getImageData') return () => ({ snapshot: true });
      return () => { canvasCalls.push(name); };
    }, set: () => true });
  }
  toDataURL() { return 'data:image/png;base64,QUJD'; }
}

const registry = {};
const document = new FakeElement('document');
document.activeElement = null;
document.getElementById = (id) => registry[id] || (registry[id] = document.appendChild(new FakeElement('div', id)));
document.createElement = (tag) => new FakeElement(tag);
document.createElementNS = (_ns, tag) => new FakeElement(tag);
global.document = document;

class HTMLTextAreaElement {}
Object.defineProperty(HTMLTextAreaElement.prototype, 'value', {
  configurable: true,
  get() { return this._value || ''; },
  set(v) { this._value = v; },
});
const textarea = new HTMLTextAreaElement();
textarea.focus = () => {}; textarea.blur = () => {};
textarea.dispatchEvent = (event) => { if (event.type === 'keydown' && event.ctrlKey && event.key === 'Enter') commits.push(textarea._value); };
const hiddenContainer = { querySelector: () => textarea };
global.Event = class { constructor(type, init) { Object.assign(this, init || {}); this.type = type; } };
global.KeyboardEvent = class extends global.Event {};
global.Image = class { set src(_v) { if (this.onload) this.onload(); } };
global.window = { HTMLTextAreaElement, parent: { document: { querySelector: () => hiddenContainer } } };

function key(element, keyName, extra) { return element.dispatch('keydown', Object.assign({ key: keyName }, extra || {})); }
module.exports = { document, commits, canvasCalls, flushTimers, key };
