import { SYMBOLS, generateGraph, validateRequest, parseRules, referenceTrace, verifySnapshot } from './core.mjs';
import { submitProxy } from './proxy.mjs';

const $ = id => document.getElementById(id);
let config = { space: null, maxSteps: 256 }, examples = [], client = null, submission = null;
let request = { mapping: generateGraph(26, 'cycle', 42), start: 'A', steps: 12 };
let activeCount = 26, frame = null, selected = 0, playing = false, timer = null, busy = false, generation = 0;
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

function text(id, value) { $(id).textContent = value; }
function status(value, error = false) { text('status', value); $('status').classList.toggle('error', error); }
function options(id, values, current) {
  $(id).replaceChildren(...values.map(value => { const option = document.createElement('option'); option.value = value; option.textContent = value; return option; }));
  $(id).value = current;
}
function visibleSymbols() {
  // A wrong prediction outside the working subset must still be visible.
  return SYMBOLS.filter((s, i) => i < activeCount || frame?.trace.some(r => r.prediction === s));
}
function editorFields() {
  const symbols = SYMBOLS.slice(0, activeCount), map = Object.fromEntries(request.mapping);
  options('start', symbols, request.start);
  options('edge-source', symbols, symbols[0]); options('edge-target', symbols, map[symbols[0]]);
  $('rule-text').value = symbols.map(s => `${s} → ${map[s]}`).join(', ');
  $('node-count').value = activeCount; $('steps').value = request.steps;
  text('graph-description', activeCount === 26 ? '26 nodes · one outgoing edge per node · cycles allowed' : `${activeCount} working nodes · remaining A–Z nodes self-loop`);
}
function clearResult() {
  generation++; pause(); frame = null; selected = 0; $('examples').value = '';
  text('source-badge', 'New graph'); render();
  status(config.space ? 'Graph ready. Run the model to see its predictions.' : 'Graph ready. Live hosting is not connected yet; recorded examples are available above.');
}
function updateRequest() {
  request = validateRequest({ ...request, start: $('start').value, steps: Number($('steps').value) }, config.maxSteps);
}
function guarded(action) { return () => { try { action(); } catch (error) { status(error.message, true); } }; }

function drawGraph() {
  const symbols = visibleSymbols(), n = symbols.length, map = Object.fromEntries(request.mapping), row = frame?.trace[selected - 1];
  const previous = selected > 1 ? frame.trace[selected - 2] : null;
  const expected = row?.expected ?? request.start, predicted = row?.prediction;
  const fromReference = previous?.expected ?? request.start, fromModel = previous?.prediction ?? request.start;
  const radius = n > 16 ? 208 : 195, nodeR = n > 20 ? 15 : 19, center = [260, 250];
  const points = Object.fromEntries(symbols.map((s, i) => { const a = i / n * Math.PI * 2 - Math.PI / 2; return [s, [center[0] + radius * Math.cos(a), center[1] + radius * Math.sin(a)]]; }));
  let edges = '';
  // Draw ordinary edges first, then the selected reference/model edges above them.
  const ordered = [...symbols].sort((a, b) => Number(a === fromReference || a === fromModel) - Number(b === fromReference || b === fromModel));
  for (const source of ordered) {
    const target = map[source]; if (!points[target]) continue;
    const a = points[source], b = points[target]; let d;
    if (source === target) {
      const dx = (a[0] - 260) / radius, dy = (a[1] - 250) / radius;
      d = `M ${a[0] - dy * 10} ${a[1] + dx * 10} C ${a[0] + dx * 48 - dy * 35} ${a[1] + dy * 48 + dx * 35}, ${a[0] + dx * 48 + dy * 35} ${a[1] + dy * 48 - dx * 35}, ${a[0] + dy * 12} ${a[1] - dx * 12}`;
    } else {
      const dx = b[0] - a[0], dy = b[1] - a[1], distance = Math.hypot(dx, dy);
      const bend = map[target] === source ? 28 : 8;
      d = `M ${a[0] + dx / distance * nodeR} ${a[1] + dy / distance * nodeR} Q ${(a[0] + b[0]) / 2 - dy / distance * bend} ${(a[1] + b[1]) / 2 + dx / distance * bend} ${b[0] - dx / distance * (nodeR + 5)} ${b[1] - dy / distance * (nodeR + 5)}`;
    }
    edges += `<path d="${d}" class="graph-edge" marker-end="url(#arrow)"/>`;
    if (row && source === fromReference) edges += `<path d="${d}" class="graph-edge active-reference" marker-end="url(#arrow-reference)"/>`;
    if (row && source === fromModel && target === predicted) edges += `<path d="${d}" class="graph-edge active-model" marker-end="url(#arrow-model)"/>`;
  }
  $('graph').innerHTML = `<defs>${[['arrow', '#b0b8ad'], ['arrow-reference', '#3c7185'], ['arrow-model', '#9c4833']].map(([id, color]) => `<marker id="${id}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="${color}"/></marker>`).join('')}<radialGradient id="center-fade"><stop offset="0%" stop-color="#fffefb"/><stop offset="65%" stop-color="#fffefb" stop-opacity=".98"/><stop offset="100%" stop-color="#fffefb" stop-opacity="0"/></radialGradient></defs>${edges}<circle cx="260" cy="250" r="112" fill="url(#center-fade)"/>${symbols.map(s => {
    const [x, y] = points[s];
    return `<g class="graph-node ${s === request.start ? 'is-start' : ''} ${s === expected ? 'is-reference' : ''} ${s === predicted ? 'is-model' : ''}" data-symbol="${s}" role="button" tabindex="0" aria-label="Node ${s}, points to ${map[s]}. Edit outgoing edge"><circle cx="${x}" cy="${y}" r="${nodeR}"/>${s === predicted ? `<circle class="model-ring" cx="${x}" cy="${y}" r="${nodeR + 5}"/>` : ''}<text x="${x}" y="${y + 4}" text-anchor="middle">${s}</text></g>`;
  }).join('')}`;
  text('graph-center-label', row ? `LOOP ${selected}` : 'STARTING AT');
  text('graph-center-value', row?.prediction ?? request.start);
  text('graph-center-detail', row ? (row.correct ? 'matches the reference' : `reference is at ${row.expected}`) : 'ready for the first loop');
}

function renderTimeline() {
  const rows = frame?.trace ?? [], length = Math.max(request.steps, rows.length);
  $('timeline').classList.toggle('dense', length > 40);
  $('timeline').replaceChildren(...Array.from({ length }, (_, i) => {
    const row = rows[i], button = document.createElement('button');
    button.className = `tick ${row ? (row.correct ? '' : 'bad') : 'pending'}`;
    button.dataset.loop = i + 1; button.disabled = !row;
    button.setAttribute('aria-pressed', String(selected === i + 1));
    button.setAttribute('aria-label', row ? `Step ${i + 1}: model ${row.prediction}, expected ${row.expected}, ${row.correct ? 'correct' : 'incorrect'}` : `Step ${i + 1}: waiting`);
    button.title = button.getAttribute('aria-label');
    const index = document.createElement('small'); index.textContent = i + 1;
    const symbol = document.createElement('span'); symbol.textContent = row?.prediction ?? '·';
    const verdict = document.createElement('small'); verdict.textContent = row ? (row.correct ? '✓' : '×') : '—';
    button.append(index, symbol, verdict); return button;
  }));
}

function metric(id, value, yes, no) {
  text(id, value == null ? '—' : value ? yes : no);
  $(id).className = value == null ? '' : value ? 'good' : 'bad';
}
function render() {
  const rows = frame?.trace ?? [], row = rows[selected - 1], summary = frame?.summary;
  drawGraph(); renderTimeline();
  text('architecture-answer', row?.prediction ?? '—'); text('expected', row?.expected ?? '—'); text('predicted', row?.prediction ?? '—');
  text('r-detail', selected ? `Loop ${selected} · h${selected - 1} → h${selected}` : 'Waiting for a loop');
  $('r-module').classList.toggle('observed', !!row);
  text('step-verdict', row ? (row.correct ? '✓ Correct' : '× Incorrect') : 'Before the first loop');
  $('step-verdict').className = `verdict ${row ? row.correct ? 'correct' : 'incorrect' : ''}`;
  text('edge-verdict', row ? (row.edge_valid ? 'Yes ✓' : 'No ×') : '—');
  text('direct-prediction', row?.direct_prediction ?? (row ? 'Not recorded' : '—'));
  text('stop-probability', row ? `${(row.stop_probability * 100).toFixed(1)}%` : '—');
  $('stop-fill').style.width = `${row ? row.stop_probability * 100 : 0}%`;
  text('controller-verdict', row ? row.stop_probability >= .5 ? 'Stop signaled · probability reached the 0.5 threshold.' : 'Continue · the learned timer has not signaled a stop.' : 'The timer decides when to stop; it does not judge whether the answer is correct.');
  text('step-count', `${selected} / ${rows.length}`); text('received', `${rows.length} predictions ${frame?.source === 'recorded' ? 'recorded' : 'received'}${busy ? ' · streaming' : ''}`);
  $('scrubber').max = rows.length; $('scrubber').value = selected; $('scrubber').disabled = !rows.length;
  $('play').disabled = !rows.length; text('play', playing ? 'Pause' : selected === rows.length && rows.length ? 'Replay' : 'Play');
  $('previous').disabled = selected === 0; $('next').disabled = selected >= rows.length;
  $('first-error').disabled = !rows.some(r => !r.correct); $('export-trace').disabled = !rows.length;
  metric('metric-answer', summary?.answer_correct, 'Correct ✓', 'Incorrect ×');
  metric('metric-trajectory', summary?.complete_trajectory, 'All steps correct ✓', 'Contains errors / missing steps');
  metric('metric-stop', summary?.exact_stop, 'Exactly on time ✓', summary?.stop_reason === 'cap' ? 'Safety cap reached' : 'Early or late');
  text('metric-error', summary ? summary.first_error ?? 'None' : '—');
  text('run-details', frame?.status === 'complete' ? `${frame.source === 'recorded' ? 'Recorded model run' : frame.served_from_cache ? 'Cached model run' : 'Live model run'} · ${rows.length} loops · ${summary.stop_reason === 'head' ? 'learned stop' : 'safety cap'}${frame.inference_seconds != null ? ` · ${frame.inference_seconds.toFixed(2)} s including observation/export overhead` : ''}` : 'Reference answers are calculated separately and never fed to the model.');
}
function pause() { playing = false; clearTimeout(timer); timer = null; }
function advance() {
  if (!playing) return;
  if (selected < (frame?.trace.length ?? 0)) selected++;
  else if (!busy) pause();
  render();
  if (playing) timer = setTimeout(advance, Number($('speed').value));
}
function selectLoop(value) { pause(); selected = Math.min(Math.max(0, value), frame?.trace.length ?? 0); render(); }
function setBusy(value) {
  busy = value;
  for (const id of ['generate','graph-type','node-count','seed','start','steps','examples','save-edge','apply-rules','edge-source','edge-target','rule-text']) $(id).disabled = value;
  $('run').disabled = value || !config.space; $('cancel').hidden = !value;
}
function loadExample(index) {
  const example = examples[index]; if (!example) return;
  pause(); request = structuredClone(example.request); activeCount = 26; selected = 0;
  frame = verifySnapshot(structuredClone(example), request); editorFields(); text('source-badge', 'Recorded run'); render();
  status(`${example.label}. This is a saved, audited model run. Press Play to inspect it.`);
}

async function runModel() {
  try { updateRequest(); } catch (error) { status(error.message, true); return; }
  if (!config.space) { status('The live service is not connected yet. Choose a recorded example to explore.', true); return; }
  clearResult(); const currentGeneration = generation; const submittedRequest = structuredClone(request);
  setBusy(true); text('source-badge', 'Live model'); status('Connecting to Hugging Face. A sleeping Space may need time to wake up…');
  try {
    if (!config.proxyUrl && !client) { const { Client } = await import(config.clientModule); client = await Client.connect(config.space, { events: ['data', 'status'] }); }
    if (currentGeneration !== generation) return;
    submission = config.proxyUrl ? submitProxy(config.proxyUrl, submittedRequest) : client.submit('/predict', { request: submittedRequest });
    for await (const message of submission) {
      if (currentGeneration !== generation) break;
      if (message.type === 'status') {
        if (message.stage === 'error') throw Error(message.message || 'The model service returned an error.');
        if (message.stage === 'pending') status(`Waiting for GPU${message.position != null ? ` · queue position ${message.position + 1}` : ''}. Recorded examples are available when the queue is busy.`);
        if (message.stage === 'generating') status('The model is running. Predictions appear as they arrive.');
      }
      if (message.type === 'data') {
        const next = verifySnapshot(message.data[0], submittedRequest);
        if (next.trace.length < (frame?.trace.length ?? 0)) throw Error('The stream moved backwards. Please retry.');
        const first = !frame; frame = next;
        if (frame.served_from_cache) text('source-badge', 'Cached model run');
        if (first && !reducedMotion) { playing = true; timer = setTimeout(advance, Number($('speed').value)); }
        render();
        if (frame.status === 'complete') {
          status(`${frame.served_from_cache ? 'Replayed a cached model run' : 'Inference complete'}: ${frame.trace.length} predictions. Playback runs at your chosen pace.`);
          break;
        }
      }
    }
    if (currentGeneration === generation && frame?.status !== 'complete') throw Error('The connection ended before the final result. Partial predictions are retained; retry to complete the run.');
  } catch (error) {
    if (currentGeneration === generation) status(`${error.message} You can still explore the recorded examples.`, true);
  } finally {
    if (currentGeneration === generation) { submission = null; setBusy(false); render(); }
  }
}

function download(name, object) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(object, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a'); link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$('generate').addEventListener('click', guarded(() => {
  const count = Number($('node-count').value), pairs = generateGraph(count, $('graph-type').value, Number($('seed').value));
  request = validateRequest({ mapping: pairs, start: SYMBOLS.slice(0, count).includes($('start').value) ? $('start').value : 'A', steps: Number($('steps').value) }, config.maxSteps);
  activeCount = count; editorFields(); clearResult();
}));
for (const id of ['start', 'steps']) $(id).addEventListener('change', guarded(() => { updateRequest(); clearResult(); }));
$('examples').addEventListener('change', () => { if ($('examples').value !== '') loadExample(Number($('examples').value)); });
$('edit-toggle').addEventListener('click', () => { const open = $('editor').hidden; $('editor').hidden = !open; $('edit-toggle').setAttribute('aria-expanded', String(open)); });
$('edge-source').addEventListener('change', () => { $('edge-target').value = Object.fromEntries(request.mapping)[$('edge-source').value]; });
$('save-edge').addEventListener('click', guarded(() => { const source = $('edge-source').value, target = $('edge-target').value; request.mapping = request.mapping.map(([a,b]) => [a, a === source ? target : b]); editorFields(); clearResult(); }));
$('apply-rules').addEventListener('click', guarded(() => { request.mapping = parseRules($('rule-text').value, activeCount); editorFields(); clearResult(); }));
function editNode(event) {
  const node = event.target.closest('[data-symbol]'); if (!node || busy) return;
  if (event.type === 'keydown' && !['Enter', ' '].includes(event.key)) return;
  if (event.type === 'keydown') event.preventDefault();
  if (!SYMBOLS.slice(0, activeCount).includes(node.dataset.symbol)) return;
  $('editor').hidden = false; $('edit-toggle').setAttribute('aria-expanded', 'true');
  $('edge-source').value = node.dataset.symbol; $('edge-target').value = Object.fromEntries(request.mapping)[node.dataset.symbol]; $('edge-target').focus();
}
$('graph').addEventListener('click', editNode); $('graph').addEventListener('keydown', editNode);
$('play').addEventListener('click', () => { if (playing) pause(); else { if (selected === frame.trace.length) selected = 0; playing = true; timer = setTimeout(advance, Number($('speed').value)); } render(); });
$('previous').addEventListener('click', () => selectLoop(selected - 1)); $('next').addEventListener('click', () => selectLoop(selected + 1));
$('scrubber').addEventListener('input', () => selectLoop(Number($('scrubber').value)));
$('timeline').addEventListener('click', event => { const tick = event.target.closest('[data-loop]'); if (tick && !tick.disabled) selectLoop(Number(tick.dataset.loop)); });
$('first-error').addEventListener('click', () => selectLoop(frame.trace.find(r => !r.correct).loop));
$('export-request').addEventListener('click', guarded(() => { if (!busy) updateRequest(); download('loopformer-graph.json', request); }));
$('export-trace').addEventListener('click', () => download('loopformer-trace.json', frame));
$('run').addEventListener('click', runModel);
$('cancel').addEventListener('click', async () => {
  generation++; pause(); const job = submission; submission = null; setBusy(false); render();
  status('Run cancelled. Any predictions received so far remain available.');
  if (job) { try { await job.cancel(); } catch { status('Playback stopped; the remote run may finish before cancellation reaches it.'); } }
});

async function initialize() {
  editorFields(); render();
  try {
    const response = await fetch('./config.json'); if (!response.ok) throw Error('Could not load demo configuration.'); config = await response.json();
    if (!Number.isInteger(config.maxSteps) || config.maxSteps < 1 || config.maxSteps > 256) throw Error('Invalid demo step limit.');
    $('steps').max = config.maxSteps;
    text('connection', config.space ? 'Live inference · Hugging Face' : 'Recorded examples · live service pending');
    $('connection').classList.toggle('live', !!config.space);
    if (config.modelUrl) { $('download-model').href = config.modelUrl; text('download-model', 'Download the model ↗'); }
    for (const link of document.querySelectorAll('.topbar > a:first-child, footer > a')) {
      link.href = config.journalUrl || '/#projects';
      link.textContent = config.journalUrl ? '← LoopFormer journal' : '← Back to projects';
    }
    document.querySelector('.hero-note a').href = config.journalUrl ? `${config.journalUrl}#architecture-5` : `${config.modelUrl || 'https://github.com/MukundShankar-dev/loopformer'}#architecture-and-inference`;
    const saved = await fetch('./examples.json'); if (!saved.ok) throw Error('Could not load recorded examples.');
    examples = (await saved.json()).filter(e => e.request.steps <= config.maxSteps);
    for (const [i, example] of examples.entries()) { const option = document.createElement('option'); option.value = i; option.textContent = example.label; $('examples').append(option); }
    if (examples.length) { $('examples').value = '0'; loadExample(0); }
    else status('Create a graph to begin.');
    setBusy(false);
  } catch (error) { status(error.message, true); text('connection', 'Service unavailable'); }
}
initialize();
