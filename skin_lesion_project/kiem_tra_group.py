
import pandas as pd
from itertools import combinations
from collections import Counter
from src.prepare import _compute_dhash_fast

manifest = pd.read_csv("data/splits/manifest.csv")
group = manifest[manifest["group_id"] == "phash_2054"]

hashes = [
    _compute_dhash_fast(path)[1]
    for path in group["image_path"]
]
hashes = [h for h in hashes if h is not None]

pairs = [
    (i, j, (hashes[i] ^ hashes[j]).bit_count())
    for i, j in combinations(range(len(hashes)), 2)
]

print("So anh:", len(hashes))
print("Tong so cap:", len(pairs))

for threshold in (2, 3, 4):
    parent = list(range(len(hashes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, distance in pairs:
        if distance <= threshold:
            a, b = find(i), find(j)
            if a != b:
                parent[b] = a

    sizes = Counter(find(i) for i in range(len(hashes)))

    print(
        f"Nguong {threshold}:",
        "So nhom:", len(sizes),
        "| Kich thuoc:", sorted(sizes.values(), reverse=True)[:15]
    )
