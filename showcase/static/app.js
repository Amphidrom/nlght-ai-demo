const gallery = document.querySelector('#gallery');
const dialog = document.querySelector('#runner');
const title = document.querySelector('#runner-title');
const story = document.querySelector('#runner-story');
const caps = document.querySelector('#runner-capabilities');
const input = document.querySelector('#runner-input');
const examples = document.querySelector('#examples');
const runButton = document.querySelector('#run');
const state = document.querySelector('#run-state');
const timeline = document.querySelector('#timeline');
const result = document.querySelector('#result');
let selected = null;

const escapeHtml = value => String(value).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));

// A case can be unavailable because its workflow was never seeded. When it also
// declares optional integrations, say which ones rather than leaving a greyed
// out card with no explanation.
function requirementNote(item) {
  const needs = item.optional_integrations || [];
  if (!needs.length) return '';
  const label = needs.map(escapeHtml).join(' · ');
  return item.available
    ? `<p class="requires">Optional stack in use: ${label}</p>`
    : `<p class="requires">Needs the optional stack: ${label}</p>`;
}

async function load() {
  const [catalog, health] = await Promise.all([
    fetch('/demo/api/cases').then(r => r.json()),
    fetch('/demo/api/health').then(r => r.json()).catch(() => ({status: 'degraded'})),
  ]);
  const healthEl = document.querySelector('#health');
  healthEl.textContent = health.status === 'ok' ? 'Runtime ready' : 'Runtime needs attention';
  healthEl.classList.toggle('ok', health.status === 'ok');
  gallery.innerHTML = catalog.cases.map((item, index) => `
    <article class="case ${item.available ? '' : 'unavailable'}">
      <button data-id="${item.id}" ${item.available ? '' : 'disabled'}>
        <span class="case-number">${String(index + 1).padStart(2, '0')}</span>
        <h3>${escapeHtml(item.title)}</h3>
        <p>${escapeHtml(item.tagline)}</p>
        <div class="tags">${item.capabilities.slice(0, 4).map(c => `<span class="tag">${escapeHtml(c)}</span>`).join('')}</div>
        ${requirementNote(item)}
      </button>
    </article>`).join('');
  gallery.querySelectorAll('button').forEach(button => button.addEventListener('click', () => openCase(catalog.cases.find(c => c.id === button.dataset.id))));
}

function openCase(item) {
  selected = item;
  title.textContent = item.title;
  story.textContent = item.story;
  caps.textContent = (item.optional_integrations || []).length
    ? `${item.capabilities.join(' · ')} — requires ${item.optional_integrations.join(' and ')}`
    : item.capabilities.join(' · ');
  input.value = JSON.stringify(item.sample_input, null, 2);
  examples.innerHTML = (item.examples || []).map((example, index) => `
    <button class="example ${index === 0 ? 'selected' : ''}" data-index="${index}">
      <b>${escapeHtml(example.title)}</b>
      <span>${escapeHtml(example.description)}</span>
    </button>`).join('');
  examples.querySelectorAll('.example').forEach(button => button.addEventListener('click', () => {
    examples.querySelectorAll('.example').forEach(item => item.classList.remove('selected'));
    button.classList.add('selected');
    input.value = JSON.stringify(item.examples[Number(button.dataset.index)].input, null, 2);
    timeline.innerHTML = '';
    result.hidden = true;
    state.textContent = '';
  }));
  timeline.innerHTML = '';
  result.hidden = true;
  result.textContent = '';
  state.textContent = '';
  runButton.disabled = false;
  dialog.showModal();
}

runButton.addEventListener('click', async () => {
  let payload;
  try { payload = JSON.parse(input.value); } catch { state.textContent = 'Input is not valid JSON.'; return; }
  runButton.disabled = true;
  state.textContent = 'Starting…';
  timeline.innerHTML = '';
  result.textContent = '';
  result.hidden = true;
  const response = await fetch(`/demo/api/cases/${selected.id}/runs`, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload),
  });
  const run = await response.json();
  if (!response.ok) { state.textContent = run.detail || 'Run could not start.'; runButton.disabled = false; return; }
  state.textContent = `Running · ${run.correlation_id.slice(0, 8)}`;
  const events = new EventSource(`/demo/api/runs/${run.correlation_id}/events`);
  ['status', 'token', 'result', 'artifact', 'error'].forEach(kind => events.addEventListener(kind, event => {
    const data = JSON.parse(event.data);
    if (kind === 'token' || kind === 'result') { result.hidden = false; result.textContent += data.content; }
    if (kind !== 'token') timeline.insertAdjacentHTML('beforeend', `<div class="event"><b>${escapeHtml(kind)}</b><span>${escapeHtml(data.content || '')}</span></div>`);
  }));
  events.addEventListener('done', event => {
    const data = JSON.parse(event.data);
    state.textContent = data.status + (data.error ? ` · ${data.error}` : '');
    runButton.disabled = false;
    events.close();
  });
});

document.querySelector('.close').addEventListener('click', () => dialog.close());
load().catch(error => { gallery.textContent = `The showcase could not be loaded: ${error}`; });
