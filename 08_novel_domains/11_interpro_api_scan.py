#!/usr/bin/env python3 -u
"""
11_interpro_api_scan.py — Submit top 100 novel domain representatives to
EBI InterProScan REST API and collect results.

Usage:
  python 11_interpro_api_scan.py [--resume]

  --resume: Skip sequences that already have results in the output file.

Strategy:
  - Submit each sequence to https://www.ebi.ac.uk/Tools/services/rest/iprscan5/
  - Poll for completion (max 10 min per sequence)
  - Parse TSV results
  - Rate limit: max 25 concurrent jobs, 1 submission per second
  - Writes results progressively to allow resumption

Inputs:
  novel_domains/results/top100_for_interpro.fasta

Outputs:
  novel_domains/results/top100_interpro_results.tsv
"""

import sys
import time
import requests
from pathlib import Path
from collections import defaultdict

BASE = Path("/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/novel_domains")
RESULTS = BASE / "results"
INPUT_FASTA = RESULTS / "top100_for_interpro.fasta"
OUTPUT_TSV = RESULTS / "top100_interpro_results.tsv"

API_BASE = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"
EMAIL = "drn2@nyu.edu"

MAX_CONCURRENT = 20
POLL_INTERVAL = 15  # seconds
MAX_POLL_TIME = 600  # 10 minutes per sequence
SUBMIT_DELAY = 1.5  # seconds between submissions

def read_fasta(path):
    """Read FASTA, return list of (id, sequence) tuples."""
    sequences = []
    current_id = None
    current_seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    sequences.append((current_id, "".join(current_seq)))
                current_id = line[1:].split()[0]
                current_seq = []
            elif current_id:
                current_seq.append(line)
        if current_id:
            sequences.append((current_id, "".join(current_seq)))
    return sequences

def submit_job(sequence, title):
    """Submit a single sequence to InterProScan API. Returns job ID."""
    # Strip stop codon asterisks (common from ORF prediction)
    clean_seq = sequence.replace("*", "").strip()
    if not clean_seq:
        print(f"    WARNING: empty sequence after cleaning")
        return None
    data = {
        "email": EMAIL,
        "title": title[:60],
        "sequence": clean_seq,
        "stype": "p",
        "goterms": "true",
        "pathways": "true",
    }
    for attempt in range(3):
        try:
            resp = requests.post(f"{API_BASE}/run", data=data, timeout=30)
            if resp.status_code == 200:
                return resp.text.strip()
            elif resp.status_code == 429:
                print(f"    Rate limited, waiting 30s...")
                time.sleep(30)
            else:
                print(f"    Submit error {resp.status_code}: {resp.text[:200]}")
                time.sleep(5)
        except requests.exceptions.RequestException as e:
            print(f"    Connection error: {e}")
            time.sleep(10)
    return None

def poll_status(job_id):
    """Poll job status until finished or timeout."""
    start = time.time()
    while time.time() - start < MAX_POLL_TIME:
        try:
            resp = requests.get(f"{API_BASE}/status/{job_id}", timeout=15)
            status = resp.text.strip()
            if status == "FINISHED":
                return "FINISHED"
            elif status in ("FAILURE", "ERROR", "NOT_FOUND"):
                return status
            # RUNNING or QUEUED
            time.sleep(POLL_INTERVAL)
        except requests.exceptions.RequestException:
            time.sleep(POLL_INTERVAL)
    return "TIMEOUT"

def get_results_tsv(job_id):
    """Get TSV results for a finished job."""
    try:
        resp = requests.get(f"{API_BASE}/result/{job_id}/tsv", timeout=30)
        if resp.status_code == 200:
            return resp.text
        else:
            print(f"    Results error {resp.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"    Results connection error: {e}")
        return None

def parse_interpro_tsv(tsv_text, domain_id):
    """Parse InterProScan TSV output. Returns list of hit dicts."""
    hits = []
    if not tsv_text or not tsv_text.strip():
        return hits
    for line in tsv_text.strip().split("\n"):
        fields = line.split("\t")
        if len(fields) >= 11:
            hit = {
                "domain_id": domain_id,
                "query_id": fields[0],
                "md5": fields[1],
                "seq_length": fields[2],
                "analysis_db": fields[3],
                "analysis_accession": fields[4],
                "analysis_description": fields[5],
                "start": fields[6],
                "end": fields[7],
                "evalue": fields[8],
                "match_status": fields[9],
                "run_date": fields[10],
                "interpro_accession": fields[11] if len(fields) > 11 else "",
                "interpro_description": fields[12] if len(fields) > 12 else "",
                "go_terms": fields[13] if len(fields) > 13 else "",
                "pathways": fields[14] if len(fields) > 14 else "",
            }
            hits.append(hit)
    return hits

def main():
    resume_mode = "--resume" in sys.argv

    # Read sequences
    sequences = read_fasta(INPUT_FASTA)
    print(f"Read {len(sequences)} sequences from {INPUT_FASTA}")

    # Check for existing results if resuming
    completed_domains = set()
    if resume_mode and OUTPUT_TSV.exists():
        with open(OUTPUT_TSV) as f:
            for line in f:
                if line.startswith("domain_id"):
                    continue
                domain_id = line.split("\t")[0]
                completed_domains.add(domain_id)
        print(f"Resume mode: {len(completed_domains)} domains already done")

    # Filter to sequences not yet done
    remaining = [(sid, seq) for sid, seq in sequences if sid not in completed_domains]
    print(f"{len(remaining)} sequences to process")

    if not remaining:
        print("All sequences already processed!")
        return

    # Open output file (append in resume mode)
    mode = "a" if resume_mode and OUTPUT_TSV.exists() else "w"
    outf = open(OUTPUT_TSV, mode)
    if mode == "w":
        # Write header
        outf.write("domain_id\tquery_id\tseq_length\tanalysis_db\tanalysis_accession\t"
                    "analysis_description\tstart\tend\tevalue\tmatch_status\t"
                    "interpro_accession\tinterpro_description\tgo_terms\tpathways\n")

    # Process sequences in batches
    n_annotated = 0
    n_no_hits = 0
    n_errors = 0

    # Submit and process one at a time with rate limiting
    for i, (domain_id, sequence) in enumerate(remaining):
        print(f"\n[{i+1}/{len(remaining)}] {domain_id[:70]}...")

        # Submit
        job_id = submit_job(sequence, domain_id)
        if not job_id:
            print(f"  FAILED to submit")
            n_errors += 1
            # Write a no-hit marker so resume skips this
            outf.write(f"{domain_id}\t{domain_id}\t{len(sequence)}\t-\t-\t"
                        f"SUBMISSION_FAILED\t-\t-\t-\t-\t-\t-\t-\t-\n")
            outf.flush()
            continue

        print(f"  Job ID: {job_id}")

        # Poll for completion
        status = poll_status(job_id)
        print(f"  Status: {status}")

        if status == "FINISHED":
            tsv = get_results_tsv(job_id)
            hits = parse_interpro_tsv(tsv, domain_id)
            if hits:
                n_annotated += 1
                for h in hits:
                    outf.write(f"{h['domain_id']}\t{h['query_id']}\t{h['seq_length']}\t"
                               f"{h['analysis_db']}\t{h['analysis_accession']}\t"
                               f"{h['analysis_description']}\t{h['start']}\t{h['end']}\t"
                               f"{h['evalue']}\t{h['match_status']}\t"
                               f"{h['interpro_accession']}\t{h['interpro_description']}\t"
                               f"{h['go_terms']}\t{h['pathways']}\n")
                print(f"  {len(hits)} hits from databases: {set(h['analysis_db'] for h in hits)}")
            else:
                n_no_hits += 1
                outf.write(f"{domain_id}\t{domain_id}\t{len(sequence)}\t-\t-\t"
                           f"NO_MATCHES\t-\t-\t-\t-\t-\t-\t-\t-\n")
                print(f"  No InterPro matches")
        else:
            n_errors += 1
            outf.write(f"{domain_id}\t{domain_id}\t{len(sequence)}\t-\t-\t"
                       f"STATUS_{status}\t-\t-\t-\t-\t-\t-\t-\t-\n")
            print(f"  Error: {status}")

        outf.flush()
        time.sleep(SUBMIT_DELAY)

    outf.close()

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total processed: {len(remaining)}")
    print(f"  With InterPro hits: {n_annotated}")
    print(f"  No hits (novel): {n_no_hits}")
    print(f"  Errors: {n_errors}")
    print(f"  Output: {OUTPUT_TSV}")

if __name__ == "__main__":
    main()
