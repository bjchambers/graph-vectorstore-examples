from typing import Iterable, Optional, Tuple
import json
import zipfile
import dotenv
from langchain_core.documents import Document
from langchain_core.graph_vectorstores import GraphVectorStore
from langchain_core.graph_vectorstores.links import METADATA_LINKS_KEY, Link
from tqdm import tqdm
import concurrent.futures
from os.path import dirname, join as joinpath

from utils.batched import batched

dotenv.load_dotenv()

LINES_IN_FILE=5989847

PARA_WITH_HYPERLINK = joinpath(dirname(__file__), 'para_with_hyperlink.zip')

def documents(skip: int = 0) -> Iterable[Tuple[int, Document]]:
    with zipfile.ZipFile(PARA_WITH_HYPERLINK, 'r') as archive:
        with archive.open('para_with_hyperlink.jsonl', 'r') as para_with_hyperlink:
            lines = iter(enumerate(tqdm(para_with_hyperlink, total=LINES_IN_FILE)))
            if skip > 0:
                [next(lines, None) for _ in range(skip)]
            try:
                while True:
                    index, line = next(lines)
                    para = json.loads(line)

                    id = para["id"]
                    links = {
                        Link.outgoing(kind="href", tag=id)
                        for m in para["mentions"]
                        if m["ref_ids"] is not None
                        for id in m["ref_ids"]
                    }
                    links.add(Link.incoming(kind="href", tag=id))
                    yield index, Document(
                        id = id,
                        page_content = " ".join(para["sentences"]),
                        metadata = {
                            "content_id": para["id"],
                            METADATA_LINKS_KEY: list(links)
                        },
                    )
            except StopIteration:
                pass

# cassio.init(auto=True)

# from cassio.config import check_resolve_session
# check_resolve_session().default_timeout = 30.0

def load_batch(batch, knowledge_store: GraphVectorStore):
    if batch:
        docs = [doc for _, doc in batch]
        min_index, _ = batch[0]
        max_index, _ = batch[len(batch) - 1]
        knowledge_store.add_documents(docs)
    return (min_index, max_index)

BATCH_SIZE=1000
MAX_IN_FLIGHT=5

def load_2wikimultihop(knowledge_store: GraphVectorStore):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = set()
        for batch in batched(documents(), BATCH_SIZE):
            futures.add(executor.submit(load_batch, batch, knowledge_store))

            if len(futures) >= MAX_IN_FLIGHT:
                done, pending = concurrent.futures.wait(futures, return_when="FIRST_COMPLETED")
                for future in done:
                    (min_index, max_index) = future.result()
                    print(f"Completed batch ({min_index}, {max_index})")
                futures = pending
        while futures:
            done, pending = concurrent.futures.wait(futures, return_when="FIRST_COMPLETED")
            for future in done:
                (min_index, max_index) = future.result()
                print(f"Completed batch ({min_index}, {max_index})")
            futures = pending