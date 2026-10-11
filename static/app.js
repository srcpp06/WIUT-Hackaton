// FlowLens UI behaviour: theme toggle (remembered per browser), count-up numbers, score gauge and bars.
(function () {
  const root = document.documentElement;
  const motion = () => root.classList.contains('motion');   // set in <head>: scripts run and motion is allowed
  const effective = () => root.dataset.theme || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');

  document.addEventListener('click', (e) => {
    const b = e.target.closest('[data-theme-toggle]');
    if (!b) return;
    const next = effective() === 'dark' ? 'light' : 'dark';
    root.dataset.theme = next;
    try { localStorage.setItem('theme', next); } catch (_) { /* private mode: the choice lasts for this page */ }
    document.dispatchEvent(new CustomEvent('themechange'));
  });

  // <b data-count="95.4" data-dec="1" data-suffix="%">95.4%</b>: the server text stays correct without JS
  function countUp(el) {
    const target = parseFloat(el.dataset.count), dec = parseInt(el.dataset.dec || '0', 10);
    if (!isFinite(target) || !motion()) return;
    const fmt = new Intl.NumberFormat(root.lang || 'en', { minimumFractionDigits: dec, maximumFractionDigits: dec });
    const pre = el.dataset.prefix || '', suf = el.dataset.suffix || '', t0 = performance.now(), dur = 900;
    const step = (now) => {
      const k = Math.min(1, (now - t0) / dur), eased = 1 - Math.pow(1 - k, 3);
      el.textContent = pre + fmt.format(target * eased) + suf;
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function animate(scope) {
    if (!motion()) return;   // the server-rendered state is already final
    scope.querySelectorAll('[data-count]').forEach(countUp);
    scope.querySelectorAll('.bar-track i[data-w]').forEach((i) => {
      i.style.width = '0';
      requestAnimationFrame(() => requestAnimationFrame(() => { i.style.width = i.dataset.w; }));
    });
    scope.querySelectorAll('.gauge .val[data-p]').forEach((c) => {
      const len = 2 * Math.PI * parseFloat(c.getAttribute('r'));
      c.style.strokeDasharray = String(len);
      c.style.strokeDashoffset = String(len);
      requestAnimationFrame(() => requestAnimationFrame(() => { c.style.strokeDashoffset = String(len * (1 - parseFloat(c.dataset.p))); }));
    });
  }

  // the level filter swaps only the table, so move the highlight here
  document.addEventListener('click', (e) => {
    const a = e.target.closest('.seg a[hx-get]');
    if (!a) return;
    a.parentElement.querySelectorAll('a').forEach((x) => x.classList.toggle('on', x === a));
  });

  document.addEventListener('DOMContentLoaded', () => {
    animate(document);
    document.body.addEventListener('htmx:afterSwap', (e) => animate(e.target));
  });
})();
