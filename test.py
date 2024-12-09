import pyterrier as pt
import shutil

# clear folder
shutil.rmtree("./out/")

pt.init()  

# Indexing documents
indexer = pt.FilesIndexer("./out/")
index_ref = indexer.index(["./docs/doc1.txt", "./docs/doc2.txt"])

# Searching using BM25
bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25", verbose=True)
results = bm25.search("How is the weather in summer")

# Print results
print(results)
