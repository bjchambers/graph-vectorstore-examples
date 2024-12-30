import asyncio

import astrapy
import astrapy.exceptions
import httpx
import backoff
from typing import Iterable, Iterator, TypeVar, Callable
import json
import zipfile
import dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.graph_vectorstores.base import Link, METADATA_LINKS_KEY
from tqdm import tqdm
import concurrent.futures
from os.path import dirname, join as joinpath
from math import ceil

from utils.batched import batched
from utils.persistent_iteration import Offset, PersistentIteration

dotenv.load_dotenv()

LINES_IN_FILE=5989847

PARA_WITH_HYPERLINK = joinpath(dirname(__file__), 'para_with_hyperlink.zip')

def wikipedia_lines() -> Iterable[str]:
    with zipfile.ZipFile(PARA_WITH_HYPERLINK, 'r') as archive:
        with archive.open('para_with_hyperlink.jsonl', 'r') as para_with_hyperlink:
            for line in para_with_hyperlink:
                yield line

def parse_document(line: str) -> Document:
    para = json.loads(line)

    id = para["id"]
    links = {
        Link.outgoing(kind="href", tag=id)
        for m in para["mentions"]
        if m["ref_ids"] is not None
        for id in m["ref_ids"]
    }
    links.add(Link.incoming(kind="href", tag=id))
    return Document(
        id = id,
        page_content = " ".join(para["sentences"]),
        metadata = {
            "content_id": para["id"],
            METADATA_LINKS_KEY: list(links)
        },
    )

BATCH_SIZE=1000
MAX_IN_FLIGHT=5

EXCEPTIONS_TO_RETRY = (
    httpx.NetworkError,
    astrapy.exceptions.DataAPIException,
)

MAX_RETRIES = 8

StoreT = TypeVar("StoreT")

def prepare_batch(lines: Iterable[str]) -> Iterable[Document]:
    return [parse_document(line) for line in lines]

BatchPreparer = Callable[[Iterator[str]], Iterator[Document]]

async def aload_2wikimultihop(store: VectorStore,
                              batch_prepare: BatchPreparer = prepare_batch) -> None:
    persistence = PersistentIteration(
        journal_name="load_2wikimultihop.jrnl",
        iterator = batched(wikipedia_lines(), BATCH_SIZE)
    )
    total_batches = ceil(LINES_IN_FILE / BATCH_SIZE) - persistence.completed_count()
    if persistence.completed_count() > 0:
        print(f"Resuming loading with {persistence.completed_count()} completed, {total_batches} remaining")
    async with asyncio.TaskGroup() as tg:
        tasks = []

        @backoff.on_exception(
            backoff.expo,
            EXCEPTIONS_TO_RETRY,
            max_tries = MAX_RETRIES,
        )
        async def add_docs(batch_docs, offset) -> None:
            await store.aadd_documents(batch_docs)
            persistence.ack(offset)
        for offset, batch_lines in tqdm(persistence, total=total_batches):
            batch_docs = batch_prepare(batch_lines)
            if batch_docs:
                tasks.append(tg.create_task(add_docs(batch_docs, offset)))
                while len(tasks) >= MAX_IN_FLIGHT:
                    _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    tasks = list(pending)
            else:
                persistence.ack(offset)

    assert persistence.pending_count() == 0

def load_2wikimultihop(store: VectorStore,
                       batch_preparer: BatchPreparer = prepare_batch) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_IN_FLIGHT) as executor:
        futures = set()
        persistence = PersistentIteration(
            journal_name="load_2wikimultihop.jrnl",
            iterator = batched(wikipedia_lines(), BATCH_SIZE)
        )
        total_batches = ceil(LINES_IN_FILE / BATCH_SIZE) - persistence.completed_count()
        if persistence.completed_count() > 0:
            print(f"Resuming loading with {persistence.completed_count()} completed, {total_batches} remaining")

        @backoff.on_exception(
            backoff.expo,
            EXCEPTIONS_TO_RETRY,
            max_tries = MAX_RETRIES,
        )
        def add_docs(batch_docs, offset):
            store.add_documents(batch_docs)
            persistence.ack(offset)

        for offset, batch_lines in tqdm(persistence, total=total_batches):
            batch_docs = batch_preparer(batch_lines)
            if batch_docs:
                futures.add(executor.submit(add_docs(batch_docs, offset)))
                while len(futures) >= MAX_IN_FLIGHT:
                    done, pending = concurrent.futures.wait(futures, return_when="FIRST_COMPLETED")
                    for future in done:
                        _ = future.result()
                    futures = pending
            else:
                persistence.ack(offset)

        while futures:
            done, pending = concurrent.futures.wait(futures, return_when="ALL_COMPLETED")
            for future in done:
                _ = future.result()

            futures = pending

        assert persistence.pending_count() == 0