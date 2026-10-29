export const SYMBOLS = [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ'];

export function rng(seed) {
  let value = seed >>> 0;
  return () => { value += 0x6D2B79F5; let x = value; x = Math.imul(x ^ x >>> 15, x | 1); x ^= x + Math.imul(x ^ x >>> 7, x | 61); return ((x ^ x >>> 14) >>> 0) / 4294967296; };
}

function shuffle(values, random) {
  const result = [...values];
  for (let i = result.length - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [result[i], result[j]] = [result[j], result[i]]; }
  return result;
}

export function generateGraph(count, mode, seed) {
  if (!Number.isInteger(count) || count < 2 || count > 26) throw Error('Choose 2–26 nodes.');
  if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295) throw Error('Seed must be an integer from 0 to 4294967295.');
  if (!['cycle', 'permutation', 'random'].includes(mode)) throw Error('Unknown graph type.');
  const random = rng(seed), active = SYMBOLS.slice(0, count), mapping = Object.fromEntries(SYMBOLS.map(s => [s, s]));
  const order = shuffle(active, random);
  if (mode === 'cycle') order.forEach((s, i) => { mapping[s] = order[(i + 1) % count]; });
  if (mode === 'permutation') active.forEach((s, i) => { mapping[s] = order[i]; });
  if (mode === 'random') active.forEach(s => { mapping[s] = active[Math.floor(random() * count)]; });
  return shuffle(SYMBOLS, random).map(s => [s, mapping[s]]);
}

export function validateRequest(request, maxSteps = 256) {
  if (!request || !Array.isArray(request.mapping) || request.mapping.length !== 26) throw Error('A complete A–Z rule table is required.');
  if (request.mapping.some(p => !Array.isArray(p) || p.length !== 2 || p.some(s => !SYMBOLS.includes(s))) || new Set(request.mapping.map(p => p[0])).size !== 26) throw Error('Every A–Z node needs exactly one valid destination.');
  if (!SYMBOLS.includes(request.start)) throw Error('Choose a valid start node.');
  if (!Number.isInteger(request.steps) || request.steps < 1 || request.steps > maxSteps) throw Error(`Choose 1–${maxSteps} steps.`);
  return request;
}

export function referenceTrace(request, loops = request.steps) {
  validateRequest(request, Number.MAX_SAFE_INTEGER);
  const mapping = Object.fromEntries(request.mapping), result = []; let current = request.start;
  for (let i = 0; i < loops; i++) { current = mapping[current]; result.push(current); }
  return result;
}

export function parseRules(text, count) {
  const active = SYMBOLS.slice(0, count), pairs = [];
  const pattern = /\s*([A-Z])\s*(?:->|→|:)\s*([A-Z])\s*/;
  for (const line of text.trim().split(/[,;\n]+/)) {
    const match = line.match(pattern);
    if (!match || match[0].length !== line.length || !active.includes(match[1]) || !active.includes(match[2])) throw Error(`Use rules such as A → B, with letters A–${active.at(-1)}.`);
    pairs.push([match[1], match[2]]);
  }
  if (pairs.length !== count || new Set(pairs.map(p => p[0])).size !== count) throw Error('Include exactly one rule per visible node.');
  return [...pairs, ...SYMBOLS.slice(count).map(s => [s, s])];
}

export function verifySnapshot(frame, request) {
  if (!frame || frame.schema_version !== 1 || !['streaming', 'complete'].includes(frame.status) || !Array.isArray(frame.trace) || frame.trace.length > 272) throw Error('Unexpected response from the model service.');
  if (JSON.stringify(frame.request) !== JSON.stringify(request)) throw Error('The returned trace belongs to a different graph request.');
  const expected = referenceTrace(request, frame.trace.length), mapping = Object.fromEntries(request.mapping);
  let previous = request.start;
  for (const [i, row] of frame.trace.entries()) {
    if (row.loop !== i + 1 || !SYMBOLS.includes(row.prediction) || row.expected !== expected[i] || row.correct !== (row.prediction === expected[i]) || row.edge_valid !== (mapping[previous] === row.prediction) || !Number.isFinite(row.stop_probability) || row.stop_probability < 0 || row.stop_probability > 1) throw Error('The model trace failed the reference consistency check.');
    previous = row.prediction;
  }
  if (frame.status === 'complete') {
    const s = frame.summary, n = frame.trace.length, predictions = frame.trace.map(r => r.prediction);
    const targets = referenceTrace(request);
    if (!n || !s || !['head', 'cap'].includes(s.stop_reason) || s.executed_loops !== n || s.answer_correct !== (predictions.at(-1) === targets.at(-1)) || s.complete_trajectory !== (n >= request.steps && predictions.slice(0, request.steps).every((p, i) => p === targets[i])) || s.exact_stop !== (s.stop_reason === 'head' && n === request.steps)) throw Error('The model summary failed the reference consistency check.');
  }
  return frame;
}
