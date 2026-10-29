/** Anonymous browser → private owner-authenticated server → streamed model frames. */
export function submitProxy(url, request) {
  const controller = new AbortController();
  return {
    cancel: async () => controller.abort(),
    async *[Symbol.asyncIterator]() {
      const response = await fetch(url, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request), signal: controller.signal,
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw Error(detail.error || `The model service returned HTTP ${response.status}.`);
      }
      if (!response.body) throw Error('The model service did not return a stream.');
      const reader = response.body.getReader(), decoder = new TextDecoder();
      let pending = '';
      try {
        while (true) {
          const { value, done } = await reader.read();
          pending += decoder.decode(value, { stream: !done });
          const lines = pending.split('\n'); pending = lines.pop();
          if (pending.length > 2_000_000) throw Error('Unexpectedly large model response.');
          if (done && pending.trim()) lines.push(pending);
          for (const line of lines) {
            if (!line.trim()) continue;
            const frame = JSON.parse(line);
            if (frame.error) throw Error(frame.error);
            if (frame.proxy_status) yield { type: 'status', stage: frame.proxy_status };
            else yield { type: 'data', data: [frame] };
          }
          if (done) break;
        }
      } finally {
        await reader.cancel().catch(() => {});
        reader.releaseLock();
      }
    },
  };
}
