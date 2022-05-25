import faiss

def build_flat_index(feature_vectors, ids, dim=128):
    # res = faiss.StandardGpuResources()
    # try:
    #     index = faiss.GpuIndexFlatL2(res, dim)
    # except:
    index = faiss.IndexFlatL2(dim)
    id_mapped_index = faiss.IndexIDMap(index)
    id_mapped_index.add_with_ids(feature_vectors, ids)
    return id_mapped_index

def find_ground_truth(query_feature_vectors, index, recall=100):
    return index.search(query_feature_vectors, recall)
