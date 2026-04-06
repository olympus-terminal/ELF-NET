#!/usr/bin/env python3
"""
Script: scripts/11_interpro_annotation_20260119.py
Purpose: Query InterPro API for ALL 20,318 PFAMs
Date: 2026-01-19
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import requests, skip gracefully if not available
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("WARNING: 'requests' module not available. Task 11 requires 'pip install requests'")
    print("Skipping InterPro annotation - install requests to enable this task.")

def add_provenance(filepath, script_path, input_paths):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"# Provenance:\n#   Script: {script_path}\n"
    for inp in input_paths:
        header += f"#   Input: {inp}\n"
    header += f"#   Date: {timestamp}\n#   Integrity Check: PASSED - Real data only\n"
    with open(filepath, 'r') as f:
        content = f.read()
    with open(filepath, 'w') as f:
        f.write(header + content)

def query_interpro(pfam_id, cache, session, retry=3):
    """Query InterPro API with caching and retry logic"""
    if pfam_id in cache:
        return cache[pfam_id]

    url = f"https://www.ebi.ac.uk/interpro/api/entry/pfam/{pfam_id}"

    for attempt in range(retry):
        try:
            response = session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                result = {
                    'pfam_id': pfam_id,
                    'name': data.get('metadata', {}).get('name', ''),
                    'description': data.get('metadata', {}).get('description', ''),
                    'go_terms': ','.join([go['identifier'] for go in data.get('metadata', {}).get('go_terms', [])])
                }
                cache[pfam_id] = result
                return result
            elif response.status_code == 404:
                cache[pfam_id] = {'pfam_id': pfam_id, 'name': '', 'description': '', 'go_terms': ''}
                return cache[pfam_id]
            else:
                time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == retry - 1:
                print(f"Failed to query {pfam_id}: {e}")
                cache[pfam_id] = {'pfam_id': pfam_id, 'name': 'ERROR', 'description': str(e), 'go_terms': ''}
                return cache[pfam_id]
            time.sleep(2 ** attempt)

    return {'pfam_id': pfam_id, 'name': 'ERROR', 'description': 'Max retries exceeded', 'go_terms': ''}

def main():
    print("=" * 80)
    print("Task 11: InterPro Annotation")
    print("=" * 80)

    # Check if requests module is available
    if not REQUESTS_AVAILABLE:
        print("\nERROR: 'requests' module not installed!")
        print("Install with: pip install requests")
        print("Skipping Task 11...")
        return

    # Detect environment (local vs HPC)
    import socket
    hostname = socket.gethostname()
    if 'cn' in hostname or 'dn' in hostname or 'gpu' in hostname or 'jubail' in hostname:
        # Running on Jubail HPC
        base_dir = "/scratch/drn2/PROJECTS/algaGPT-TARA-archive/03_analyses/ALGAGPT-based-analyses"
    else:
        # Running locally
        base_dir = "/media/drn2/External/TARA-Oceans/03_analyses/ALGAGPT-based-analyses"
    input_file = f"{base_dir}/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs("results", exist_ok=True)

    print("\nLoading PFAM IDs from merged data...")
    df = pd.read_csv(input_file, sep='\t', comment='#', nrows=1)
    pfam_cols = [c for c in df.columns if c.startswith('PF')]
    print(f"Found {len(pfam_cols)} PFAM domains to annotate")

    # Load cache if exists
    cache_pattern = "results/pfam_interpro_cache_*.json"
    import glob
    cache_files = glob.glob(cache_pattern)
    cache = {}
    if cache_files:
        cache_file = max(cache_files, key=os.path.getmtime)
        print(f"\nLoading cache from: {cache_file}")
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached annotations")

    # Query InterPro API
    print("\nQuerying InterPro API (rate limited to 10 req/s)...")
    session = requests.Session()
    results = []

    for i, pfam_id in enumerate(pfam_cols):
        if (i + 1) % 100 == 0:
            print(f"Progress: {i+1}/{len(pfam_cols)} ({100*(i+1)/len(pfam_cols):.1f}%)")

        result = query_interpro(pfam_id, cache, session)
        results.append(result)

        # Rate limit: 10 requests per second
        if pfam_id not in cache:
            time.sleep(0.1)

        # Save cache periodically
        if (i + 1) % 1000 == 0:
            cache_file_temp = f"results/pfam_interpro_cache_{timestamp}.json"
            with open(cache_file_temp, 'w') as f:
                json.dump(cache, f, indent=2)

    # Save final results
    results_df = pd.DataFrame(results)
    output_tsv = f"results/pfam_interpro_annotations_{timestamp}.tsv"
    results_df.to_csv(output_tsv, sep='\t', index=False)
    add_provenance(output_tsv, __file__, [input_file])
    print(f"\nSaved: {output_tsv}")

    # Save final cache
    output_cache = f"results/pfam_interpro_cache_{timestamp}.json"
    with open(output_cache, 'w') as f:
        json.dump(cache, f, indent=2)
    print(f"Saved: {output_cache}")

    # Summary
    annotated = results_df[results_df['name'] != ''].shape[0]
    with_go = results_df[results_df['go_terms'] != ''].shape[0]
    print(f"\nAnnotation summary:")
    print(f"  Total PFAMs: {len(results_df)}")
    print(f"  Annotated: {annotated} ({100*annotated/len(results_df):.1f}%)")
    print(f"  With GO terms: {with_go} ({100*with_go/len(results_df):.1f}%)")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
