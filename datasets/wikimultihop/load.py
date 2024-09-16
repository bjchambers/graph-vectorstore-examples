from typing import Iterable, Iterator
import json
import zipfile
import dotenv
from langchain_core.documents import Document
from langchain_core.graph_vectorstores import GraphVectorStore
from langchain_core.graph_vectorstores.links import METADATA_LINKS_KEY, Link
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

def load_batch(offset: Offset,
               lines: Iterator[str],
               knowledge_store: GraphVectorStore,
               persistence: PersistentIteration[Iterator[str]]):
    docs = [parse_document(line) for line in lines]
    if docs:
        knowledge_store.add_documents(docs)
    persistence.ack(offset)

BATCH_SIZE=1000
MAX_IN_FLIGHT=5

def load_2wikimultihop(knowledge_store: GraphVectorStore):
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_IN_FLIGHT) as executor:
        futures = set()
        persistence = PersistentIteration(
            journal_name="load_2wikimultihop.jrnl",
            iterator = batched(wikipedia_lines(), BATCH_SIZE)
        )
        total_batches = ceil(LINES_IN_FILE / BATCH_SIZE) - persistence.completed_count()
        if persistence.completed_count() > 0:
            print(f"Resuming loading with {persistence.completed_count()} completed, {total_batches} remaining")
        for offset, batch in tqdm(persistence, total=total_batches):
            futures.add(executor.submit(load_batch, offset, batch, knowledge_store, persistence))

            if len(futures) >= MAX_IN_FLIGHT:
                done, pending = concurrent.futures.wait(futures, return_when="FIRST_COMPLETED")
                for future in done:
                    _ = future.result()
                futures = pending

        while futures:
            done, pending = concurrent.futures.wait(futures, return_when="ALL_COMPLETED")
            for future in done:
                _ = future.result()
            futures = pending

        assert persistence.pending_count() == 0