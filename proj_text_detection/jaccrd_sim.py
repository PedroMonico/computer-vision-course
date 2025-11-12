def jaccard_similarity(sentence1, sentence2):
    # Tokenize senteces into set of words
    set1 = set(sentence1.lower().split())
    set2 = set(sentence2.lower().split())

    # Calculate Jacard similarity
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))

    # Avoid division by zero if both sets are empty
    similarity = intersection_size / union_size if union_size != 0 else 0.0

    return similarity