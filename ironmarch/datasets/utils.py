def chunk_list(list_in, chunk_size):
    chunks = []
    num_chunks = len(list_in) // chunk_size
    for i in range(num_chunks):
        chunks.append(list_in[i*chunk_size:i*chunk_size+chunk_size])
    chunks.append(list_in[(num_chunks)*chunk_size:])
    return chunks