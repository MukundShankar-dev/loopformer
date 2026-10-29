"""Bounded owner-authenticated streaming relay; never expose the owner token."""

import asyncio
from collections import OrderedDict, deque
import json
from time import monotonic
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse


QUOTA_MESSAGE = ("The demo's shared Hugging Face allowance is used up. "
                 "Please try later or explore a recorded run. No sign-in is required.")


def create_proxy(validate: Callable[[Any], dict], client_factory: Callable, *,
                 origins: tuple[str, ...], cooldown: float = 10,
                 hourly_limit: int = 20, timeout: float = 120) -> FastAPI:
    """One process/replica. Cache only complete model results, preserving rule order.

    Origin checks limit browser embedding, not determined abuse. Global admission
    limits bound owner usage even when callers forge Origin. HF retains its quota.
    Factory creates the authenticated Gradio client privately on the server.
    """
    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    api.add_middleware(CORSMiddleware, allow_origins=list(origins),
                       allow_methods=["POST"], allow_headers=["Content-Type"])
    cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()
    starts: deque[float] = deque()
    active = False
    blocked_until = 0.0
    client = None

    @api.post('/api/predict')
    async def predict(request: Request):
        nonlocal active, blocked_until, client
        if request.headers.get('origin') not in origins:
            return JSONResponse({'error': 'This endpoint serves the LoopFormer playground.'}, 403)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 4096:
                return JSONResponse({'error': 'Graph request is too large.'}, 413)
        try:
            payload = validate(json.loads(body))
        except (ValueError, TypeError, UnicodeDecodeError):
            return JSONResponse({'error': 'Supply a valid A–Z graph, start node and 1–256 steps.'}, 400)
        key = json.dumps(payload, separators=(',', ':'))
        now = monotonic()
        if key in cache:
            saved_at, frame = cache[key]
            if now - saved_at < 3600:
                cache.move_to_end(key)
                return StreamingResponse(iter([json.dumps({**frame, 'served_from_cache': True}) + '\n']),
                                         media_type='application/x-ndjson',
                                         headers={'Cache-Control': 'no-store'})
            del cache[key]
        while starts and now - starts[0] >= 3600:
            starts.popleft()
        if active or now < blocked_until or len(starts) >= hourly_limit:
            delay = max(1, int(blocked_until - now + 1))
            if len(starts) >= hourly_limit:
                delay = max(delay, int(starts[0] + 3600 - now + 1))
            message = ('The demo is serving another run or resting its shared allowance. '
                       'Try later, replay a previous graph, or choose a recorded example.')
            return JSONResponse({'error': message}, 429, headers={'Retry-After': str(delay)})
        # No await between admission and reservation: concurrent requests cannot race.
        active = True
        starts.append(now)
        blocked_until = now + cooldown

        async def stream():
            nonlocal active, blocked_until, client
            job = None
            finished = False
            try:
                yield json.dumps({'proxy_status': 'pending'}) + '\n'
                if client is None:
                    client = await asyncio.to_thread(client_factory)
                job = await asyncio.to_thread(client.submit, payload, api_name='/predict')
                cursor = 0
                heartbeat = monotonic()
                while True:
                    if monotonic() - now > timeout:
                        raise TimeoutError('Model request timed out')
                    if await request.is_disconnected():
                        return
                    done = job.done()
                    outputs = job.outputs()
                    for frame in outputs[cursor:]:
                        if not isinstance(frame, dict) or frame.get('request') != payload:
                            raise ValueError('Unexpected model response')
                        if frame.get('status') == 'complete':
                            cache[key] = (monotonic(), frame)
                            while len(cache) > 32:
                                cache.popitem(last=False)
                            finished = True
                        yield json.dumps(frame) + '\n'
                    cursor = len(outputs)
                    if done:
                        final = job.result()
                        if not isinstance(final, dict) or final.get('status') != 'complete' or final.get('request') != payload:
                            raise ValueError('Model did not complete')
                        cache[key] = (monotonic(), final)
                        while len(cache) > 32:
                            cache.popitem(last=False)
                        finished = True
                        # Ensure completion survives output-buffer timing differences.
                        if not outputs or outputs[-1].get('status') != 'complete':
                            yield json.dumps(final) + '\n'
                        return
                    if monotonic() - heartbeat > 5:
                        yield json.dumps({'proxy_status': 'pending'}) + '\n'
                        heartbeat = monotonic()
                    await asyncio.sleep(0.04)
            except Exception as error:
                # This is a public transport boundary. Never echo exception strings,
                # headers, tokens or upstream URLs; retain an explicit failed run.
                quota = any(word in str(error).lower() for word in ('quota', 'runs limit'))
                if quota:
                    blocked_until = monotonic() + 300
                message = QUOTA_MESSAGE if quota else 'The model service could not finish this run. Please try later.'
                yield json.dumps({'error': message}) + '\n'
            finally:
                if job is not None and not finished:
                    # Keep cancellation off the event loop; the native runner also
                    # has its fixed safety cap if cancellation arrives too late.
                    asyncio.create_task(asyncio.to_thread(job.cancel))
                active = False

        return StreamingResponse(stream(), media_type='application/x-ndjson',
                                 headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})

    return api
