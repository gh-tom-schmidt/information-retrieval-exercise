from tira.third_party_integrations import ensure_pyterrier_is_loaded
from tira.rest_api_client import Client

ensure_pyterrier_is_loaded()
tira = Client()

from pyterrier import get_dataset, IterDictIndexer, BatchRetrieve
import pyterrier_doc2query

pt_dataset = get_dataset('irds:ir-lab-wise-2024/subsampled-ms-marco-deep-learning-20241201-training')

doc2query = pyterrier_doc2query.Doc2Query(append=True) # append generated queries to the orignal document text

## >> = custom operator
indexer = doc2query >> IterDictIndexer(
    # Store the index in the `index` directory.
    "./data/index",
    meta={'docno': 50, 'text': 4096},
    # If an index already exists there, then overwrite it.
    overwrite=True,
)

index = indexer.index(pt_dataset.get_corpus_iter())

bm25 = BatchRetrieve(index, wmodel="BM25")

run = bm25(pt_dataset.get_topics('text'))