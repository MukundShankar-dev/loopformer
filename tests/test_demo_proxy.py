"""Transport contracts for the owner-funded proxy; never require real credentials."""
import asyncio
import json
from copy import deepcopy

import httpx
import pytest

from scripts.demo.proxy import create_proxy

ORIGIN = 'https://mukundshankar-dev.github.io'
PAYLOAD = {'mapping': [[s, s] for s in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'], 'start': 'A', 'steps': 12}


class Job:
    def __init__(self, payload, fail=None):
        self.frame = {'request': deepcopy(payload), 'status': 'complete', 'trace': [1]}
        self.fail = fail
        self.cancelled = False

    def done(self):
        return True

    def outputs(self):
        return [] if self.fail else [self.frame]

    def result(self):
        if self.fail:
            raise RuntimeError(self.fail)
        return self.frame

    def cancel(self):
        self.cancelled = True


class Client:
    def __init__(self, fail=None):
        self.calls = 0
        self.fail = fail

    def submit(self, payload, api_name):
        assert api_name == '/predict'
        self.calls += 1
        return Job(payload, self.fail)


def validate(value):
    if not isinstance(value, dict) or set(value) != set(PAYLOAD):
        raise ValueError('Invalid request')
    return value


def test_cached_completion_avoids_upstream_and_preserves_prompt_order():
    async def run():
        upstream = Client()
        app = create_proxy(validate, lambda: upstream, origins=(ORIGIN,), cooldown=0)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test', headers={'Origin': ORIGIN}) as client:
            first = await client.post('/api/predict', json=PAYLOAD)
            frames = [json.loads(x) for x in first.text.splitlines()]
            assert frames[0]['proxy_status'] == 'pending'
            assert frames[-1]['status'] == 'complete'
            second = await client.post('/api/predict', json=PAYLOAD)
            assert second.json()['served_from_cache']
            assert upstream.calls == 1
            reordered = {**PAYLOAD, 'mapping': list(reversed(PAYLOAD['mapping']))}
            await client.post('/api/predict', json=reordered)
            assert upstream.calls == 2
    asyncio.run(run())


def test_admission_validates_origin_size_payload_and_throttles_new_calls():
    async def run():
        upstream = Client()
        app = create_proxy(validate, lambda: upstream, origins=(ORIGIN,))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            assert (await client.post('/api/predict', json=PAYLOAD)).status_code == 403
            headers = {'Origin': ORIGIN}
            assert (await client.post('/api/predict', content='x'*4097, headers=headers)).status_code == 413
            assert (await client.post('/api/predict', json={}, headers=headers)).status_code == 400
            await client.post('/api/predict', json=PAYLOAD, headers=headers)
            changed = {**PAYLOAD, 'steps': 13}
            assert (await client.post('/api/predict', json=changed, headers=headers)).status_code == 429
            assert (await client.post('/api/predict', json=PAYLOAD, headers=headers)).json()['served_from_cache']
            assert upstream.calls == 1
    asyncio.run(run())


@pytest.mark.parametrize('message', ['secret-token in an arbitrary error', 'ZeroGPU runs limit secret-token'])
def test_failure_redacts_upstream_details_and_does_not_cache(message):
    async def run():
        upstream = Client(message)
        app = create_proxy(validate, lambda: upstream, origins=(ORIGIN,), cooldown=0)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test', headers={'Origin': ORIGIN}) as client:
            response = await client.post('/api/predict', json=PAYLOAD)
            assert 'secret-token' not in response.text
            assert 'error' in json.loads(response.text.splitlines()[-1])
            second = await client.post('/api/predict', json=PAYLOAD)
            if 'runs limit' in message:
                assert second.status_code == 429
                assert upstream.calls == 1
            else:
                assert upstream.calls == 2
    asyncio.run(run())


def test_hourly_budget_blocks_distinct_jobs_and_cors_is_restricted():
    async def run():
        upstream = Client()
        app = create_proxy(validate, lambda: upstream, origins=(ORIGIN,), cooldown=0, hourly_limit=1)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test', headers={'Origin': ORIGIN}) as client:
            response = await client.post('/api/predict', json=PAYLOAD)
            assert response.headers['access-control-allow-origin'] == ORIGIN
            response = await client.post('/api/predict', json={**PAYLOAD, 'steps': 13})
            assert response.status_code == 429
            assert int(response.headers['retry-after']) > 3500
            assert upstream.calls == 1
    asyncio.run(run())


def test_disconnect_cancels_upstream_and_releases_admission():
    async def run():
        class PendingJob(Job):
            def done(self):
                return False
            def outputs(self):
                return []
        job = PendingJob(PAYLOAD)
        class PendingClient:
            def submit(self, *args, **kwargs):
                return job
        class Request:
            headers = {'origin': ORIGIN}
            async def stream(self):
                yield json.dumps(PAYLOAD).encode()
            async def is_disconnected(self):
                return True
        app = create_proxy(validate, PendingClient, origins=(ORIGIN,), cooldown=0)
        endpoint = next(route.endpoint for route in app.routes if route.path == '/api/predict')
        response = await endpoint(Request())
        assert 'pending' in await anext(response.body_iterator)
        with pytest.raises(StopAsyncIteration):
            await anext(response.body_iterator)
        for _ in range(20):
            if job.cancelled:
                break
            await asyncio.sleep(.01)
        assert job.cancelled
        # A disconnected client does not permanently occupy the only slot.
        second = await endpoint(Request())
        assert isinstance(second, type(response))
        async for _ in second.body_iterator:
            pass
    asyncio.run(run())
