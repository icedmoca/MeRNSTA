import time
import random
import string
import sys

try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_available = False

from storage.memory_log import MemoryLog, TripletFact

def random_word(length=6):
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def random_fact(fact_id):
    subject = random_word(random.randint(4, 8))
    predicate = random.choice([
        'likes', 'hates', 'loves', 'dislikes', 'enjoys', 'prefers', 'avoids', 'adores', 'detests', 'fears', 'needs', 'wants', 'despises', 'cherishes'
    ])
    obj = random_word(random.randint(4, 8))
    confidence = round(random.uniform(0.5, 1.0), 3)
    volatility_score = round(random.uniform(0.0, 0.5), 3)
    contradiction_score = round(random.uniform(0.0, 0.2), 3)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    return TripletFact(
        id=fact_id,
        subject=subject,
        predicate=predicate,
        object=obj,
        frequency=random.randint(1, 5),
        timestamp=timestamp,
        confidence=confidence,
        contradiction_score=contradiction_score,
        volatility_score=volatility_score
    )

def print_cluster_stats(clusters, label):
    print(f"\n{label} clusters: {len(clusters)}")
    if not clusters:
        return
    sizes = [c['cluster_size'] for c in clusters]
    print(f"  Min size: {min(sizes)}  Max size: {max(sizes)}  Mean size: {sum(sizes)//len(sizes)}")
    print(f"  Top 5 clusters by size:")
    for c in sorted(clusters, key=lambda x: -x['cluster_size'])[:5]:
        print(f"    - Subject: {c['subject']}  Size: {c['cluster_size']}  Sample: {c['sample_facts']}")

def main():
    N = 100_000
    print(f"Benchmarking MemoryLog with {N} random facts...")
    if psutil_available:
        process = psutil.Process()
        mem_start = process.memory_info().rss
    
    memlog = MemoryLog(db_path='/tmp/benchmark_memlog.db')
    memlog.init_database()
    facts = [random_fact(i) for i in range(N)]
    print("Inserting facts...")
    t0 = time.perf_counter()
    # Insert in batches for speed
    batch_size = 1000
    for i in range(0, N, batch_size):
        # Convert TripletFact objects to tuples for store_triplets
        fact_tuples = [(fact.subject, fact.predicate, fact.object, fact.confidence) for fact in facts[i:i+batch_size]]
        memlog.store_triplets(fact_tuples)
    t1 = time.perf_counter()
    print(f"Inserted {N} facts in {t1-t0:.2f}s")
    if psutil_available:
        mem_mid = process.memory_info().rss
    
    clusters_before = memlog.list_clusters()
    print_cluster_stats(clusters_before, "Initial")
    
    print("Consolidating facts (clustering/compression)...")
    t2 = time.perf_counter()
    memlog.consolidate_facts()
    t3 = time.perf_counter()
    if psutil_available:
        mem_end = process.memory_info().rss
    
    clusters_after = memlog.list_clusters()
    print_cluster_stats(clusters_after, "Post-consolidation")
    
    print("\n--- Benchmark Summary ---")
    print("facts_inserted:", N)
    print("initial_clusters:", len(clusters_before))
    print("post_consolidation_clusters:", len(clusters_after))
    print(f"time_to_insert: {t1-t0:.2f}s")
    print(f"time_to_consolidate: {t3-t2:.2f}s")
    if psutil_available:
        print(f"peak_ram: {max(mem_start, mem_mid, mem_end)//(1024*1024)} MB")
    print("------------------------\n")

if __name__ == "__main__":
    main() 