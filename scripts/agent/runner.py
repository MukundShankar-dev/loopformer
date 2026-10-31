"""Run many agent episodes through one batched Decoder.

Each engine row holds one episode. The main thread decodes; a thread pool runs
an episode's code between turns (tools in the sandbox, tokenizing), and a pool
of processes grades submissions on the hidden tests. While one row waits on
the sandbox, the others keep decoding. Grading is in processes because it is
the heaviest Python work and, on threads, it starves the decoding thread of
the interpreter lock: every release costs up to a switch interval to regain.
A row is freed when its episode's last turn ends, before grading, and the next
job takes it; the lowest free row is used first, which keeps decode batches
short. Episodes are yielded as their grading finishes, so not in job order.
"""
import ctypes
import heapq
import multiprocessing
import signal
import sys
import time
from collections.abc import Generator, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass

from scripts.agent.decoding import Decoder, Generation
from scripts.agent.episode import Agent, Episode, EpisodeConfig, Request, grade_submission
from scripts.codetasks.problems import Problem


PR_SET_PDEATHSIG = 1  # linux/prctl.h


def _die_with_parent() -> None:
    """Grader initializer: a run that is killed would otherwise leave its graders running, orphaned."""
    if sys.platform == "linux":
        if ctypes.CDLL(None, use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")


@dataclass(frozen=True)
class Job:
    problem: Problem
    config: EpisodeConfig
    seed: int


def run_episodes(agent: Agent, decoder: Decoder, jobs: Iterable[Job], workers: int,
                 graders: int) -> Iterator[Episode]:
    """`workers` threads run episode code; `graders` processes (spawned, so none inherits CUDA) grade."""
    jobs = iter(jobs)
    free = list(range(decoder.engine.rows))
    active: dict[int, tuple[Job, Generator[Request, Generation, Episode]]] = {}
    stepping: dict[Future, int] = {}  # episode code running between turns, by row
    grading: set[Future] = set()
    exhausted = False
    spawn = multiprocessing.get_context("spawn")
    with (ThreadPoolExecutor(workers) as pool,
          ProcessPoolExecutor(graders, mp_context=spawn, initializer=_die_with_parent) as grading_pool):
        while not exhausted or active or grading:
            while free and not exhausted:
                job = next(jobs, None)
                if job is None:
                    exhausted = True
                    break
                row = heapq.heappop(free)
                decoder.open(row, job.seed)
                episode = agent.episode(job.problem, job.config, job.seed)
                active[row] = (job, episode)
                stepping[pool.submit(next, episode)] = row
            if not decoder.turns and stepping:
                started = time.perf_counter()
                wait(list(stepping), return_when=FIRST_COMPLETED)
                decoder.stats["idle_seconds"] += time.perf_counter() - started  # every row waits on episode code
            for future in [future for future in stepping if future.done()]:
                row = stepping.pop(future)
                job, episode = active[row]
                try:
                    request = future.result()
                except StopIteration as stop:
                    conversation = decoder.close(row)
                    stop.value.flops, stop.value.tokens_processed = conversation.flops, conversation.processed
                    del active[row]
                    heapq.heappush(free, row)
                    grading.add(grading_pool.submit(grade_submission, job.problem, stop.value, job.config))
                    continue
                generation = decoder.start(row, request.prompt, agent.matcher(request.tools), request.sampling)
                if generation is not None:
                    stepping[pool.submit(episode.send, generation)] = row
            for row, generation in decoder.step().items():
                stepping[pool.submit(active[row][1].send, generation)] = row
            if not decoder.turns and not stepping and grading:
                wait(grading, return_when=FIRST_COMPLETED)
            for future in [future for future in grading if future.done()]:
                grading.remove(future)
                yield future.result()
