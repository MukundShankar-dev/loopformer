import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { SYMBOLS, generateGraph, parseRules, referenceTrace, validateRequest, verifySnapshot } from './web/core.mjs';

test('seeded graph generators preserve full table and cycle/permutation contracts', () => {
  for (const count of [2, 8, 26]) for (const mode of ['cycle','permutation','random']) {
    const mapping = generateGraph(count, mode, 42);
    assert.deepEqual(mapping, generateGraph(count, mode, 42));
    const request = validateRequest({ mapping, start: 'A', steps: 256 });
    const map = Object.fromEntries(mapping);
    assert.equal(Object.keys(map).length, 26);
    SYMBOLS.slice(count).forEach(s => assert.equal(map[s], s));
    if (mode !== 'random') assert.equal(new Set(SYMBOLS.slice(0, count).map(s => map[s])).size, count);
    if (mode === 'cycle') assert.equal(new Set(referenceTrace(request, count)).size, count);
  }
});
test('rule editor rejects duplicates, missing edges and injected markup', () => {
  assert.equal(parseRules('A → B, B -> A', 2).length, 26);
  for (const text of ['A → B', 'A → B, A → A', 'A → C, B → A', '<script>A → B</script>, B → A']) assert.throws(() => parseRules(text, 2));
});
test('recorded model traces agree with independent reference and summary checks', () => {
  const examples = JSON.parse(fs.readFileSync(new URL('./web/examples.json', import.meta.url)));
  for (const example of examples) assert.equal(verifySnapshot(example, example.request), example);
  const corrupt = structuredClone(examples[0]); corrupt.trace[0].prediction = '!';
  assert.throws(() => verifySnapshot(corrupt, corrupt.request));
  const summary = structuredClone(examples[0]); summary.summary.exact_stop = false;
  assert.throws(() => verifySnapshot(summary, summary.request));
});

test('proxy transport preserves split streamed frames and reports quota failures', async () => {
  const { submitProxy } = await import('./web/proxy.mjs');
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async (_url, options) => {
      assert.equal(options.headers.Authorization, undefined);
      assert.equal(options.method, 'POST');
      const chunks = ['{"proxy_status":"pend', 'ing"}\n{"schema_version":1}\n'];
      return new Response(new ReadableStream({start(c) {
        for (const chunk of chunks) c.enqueue(new TextEncoder().encode(chunk));
        c.close();
      }}));
    };
    const result = [];
    for await (const value of submitProxy('/api/predict', {})) result.push(value);
    assert.deepEqual(result, [{type:'status',stage:'pending'}, {type:'data',data:[{schema_version:1}]}]);
    globalThis.fetch = async () => new Response('{"error":"Shared allowance exhausted"}', {status:429});
    await assert.rejects(async () => {for await (const _ of submitProxy('/api/predict', {})) {}}, /Shared allowance exhausted/);
    globalThis.fetch = async () => new Response('{"error":"Quota reached"}\n');
    await assert.rejects(async () => {for await (const _ of submitProxy('/api/predict', {})) {}}, /Quota reached/);
  } finally {globalThis.fetch = original;}
});
