#!/usr/bin/env python3
"""
Extract AlphaEarth embeddings from GEE+PFAM master file.

PURPOSE:
    Regenerate AlphaEarth embeddings using the current GEE+PFAM master file
    to achieve full sample coverage for GCCA analysis.

BACKGROUND:
    The original AlphaEarth embeddings (2,049 samples) were generated from:
        la4sr_algal_pfam_gps_20260114_110426.tsv

    The GEE+PFAM master file (2,044 samples) includes:
        - Original v7 samples
        - Additional MMETSP samples from hmmsearch (with .trinity... suffixes)

    These files diverged during processing, resulting in only 1,763 sample overlap.
    Running this script will regenerate AlphaEarth embeddings for all 2,044 samples
    in the GEE+PFAM file.

PREREQUISITES:
    - Google Earth Engine account (https://earthengine.google.com/)
    - Run 'earthengine authenticate' before first use
    - pip install earthengine-api pandas

USAGE:
    python extract_alphaearth_from_gee_pfam_20260122.py

Date: 2026-01-22
"""

import os
import sys
import time
import logging
from datetime import datetime

import pandas as pd
import ee

# =============================================================================
# Configuration
# =============================================================================

BATCH_SIZE = 100  # GEE rate limit safe batch size
SLEEP_BETWEEN_BATCHES = 2  # seconds
GEE_SCALE = 10  # 10m resolution for AlphaEarth
GEE_PROJECT = 'alien-waters-233207'  # Google Cloud project for GEE

# Input/output paths
BASE_DIR = "/media/drn2/External/TARA-Oceans"
INPUT_FILE = f"{BASE_DIR}/03_analyses/ALGAGPT-based-analyses/algagpt_gee_pfam_merged_SMART_20260119_100639.tsv"
OUTPUT_DIR = f"{BASE_DIR}/PythiaTIfreeLA4SR_TARA"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# =============================================================================
# Logging setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"{OUTPUT_DIR}/alphaearth_extraction_gee_pfam_{TIMESTAMP}.log")
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Functions
# =============================================================================

def load_coordinates(input_file):
    """Load GPS coordinates from GEE+PFAM file."""
    logger.info(f"Loading coordinates from: {input_file}")

    # Read file, skip comment lines
    df = pd.read_csv(input_file, sep='\t', comment='#')

    # Validate required columns
    required = ['assembly_id', 'latitude', 'longitude']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Extract only what we need for GEE
    coords_df = df[['assembly_id', 'latitude', 'longitude']].copy()

    # Validate coordinates
    valid_lat = coords_df['latitude'].between(-90, 90)
    valid_lon = coords_df['longitude'].between(-180, 180)
    valid_coords = valid_lat & valid_lon & coords_df['latitude'].notna() & coords_df['longitude'].notna()

    n_invalid = (~valid_coords).sum()
    if n_invalid > 0:
        logger.warning(f"Excluding {n_invalid} samples with invalid/missing coordinates")
        coords_df = coords_df[valid_coords].copy()

    logger.info(f"Loaded {len(coords_df)} samples with valid GPS coordinates")

    # Show ID breakdown
    mmetsp_count = coords_df['assembly_id'].str.startswith('MMETSP').sum()
    mgya_count = coords_df['assembly_id'].str.startswith('MGYA').sum()
    other_count = len(coords_df) - mmetsp_count - mgya_count
    logger.info(f"  MMETSP: {mmetsp_count}, MGYA: {mgya_count}, Other: {other_count}")

    return coords_df

def initialize_gee():
    """Initialize Google Earth Engine."""
    logger.info("Initializing Google Earth Engine...")
    try:
        ee.Initialize(project=GEE_PROJECT)
        logger.info(f"GEE initialized successfully with project: {GEE_PROJECT}")
    except Exception as e:
        logger.error(f"GEE initialization failed: {e}")
        logger.error("Run 'python3 -c \"import ee; ee.Authenticate()\"' to set up credentials")
        raise

def extract_embeddings_batch(coords_batch, mosaic):
    """
    Extract AlphaEarth embeddings for a batch of coordinates.

    Args:
        coords_batch: DataFrame with assembly_id, latitude, longitude
        mosaic: GEE mosaic image

    Returns:
        List of dicts with assembly_id and embedding values
    """
    # Create feature collection from coordinates
    features = []
    for _, row in coords_batch.iterrows():
        point = ee.Geometry.Point([row['longitude'], row['latitude']])
        features.append(ee.Feature(point, {'assembly_id': row['assembly_id']}))

    fc = ee.FeatureCollection(features)

    # Extract embeddings
    results = mosaic.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=GEE_SCALE
    )

    # Get results
    result_list = results.getInfo()['features']

    # Parse results
    batch_results = []
    for feat in result_list:
        props = feat['properties']
        record = {'assembly_id': props['assembly_id']}

        # Extract embedding dimensions (A00-A63 in the GEE output)
        for i in range(64):
            dim_name = f'A{i:02d}'  # A00, A01, ..., A63
            record[dim_name] = props.get(dim_name, None)

        batch_results.append(record)

    return batch_results

def extract_all_embeddings(coords_df):
    """
    Extract embeddings for all coordinates with batching.

    Args:
        coords_df: DataFrame with assembly_id, latitude, longitude

    Returns:
        DataFrame with assembly_id and 64 embedding dimensions
    """
    logger.info("Loading AlphaEarth image collection...")
    collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    mosaic = collection.mosaic()

    all_results = []
    n_samples = len(coords_df)
    n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE

    logger.info(f"Processing {n_samples} samples in {n_batches} batches (batch size: {BATCH_SIZE})")

    for batch_idx in range(n_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, n_samples)

        batch_df = coords_df.iloc[start_idx:end_idx]

        logger.info(f"Processing batch {batch_idx + 1}/{n_batches} (samples {start_idx + 1}-{end_idx})")

        try:
            batch_results = extract_embeddings_batch(batch_df, mosaic)
            all_results.extend(batch_results)
            logger.info(f"  Batch {batch_idx + 1} complete: {len(batch_results)} embeddings extracted")
        except Exception as e:
            logger.error(f"  Batch {batch_idx + 1} failed: {e}")
            # Add failed samples with NaN embeddings
            for _, row in batch_df.iterrows():
                record = {'assembly_id': row['assembly_id']}
                for i in range(64):
                    record[f'A{i:02d}'] = None
                all_results.append(record)
            logger.warning(f"  Added {len(batch_df)} samples with NaN embeddings")

        # Rate limiting
        if batch_idx < n_batches - 1:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    return results_df

def validate_embeddings(embeddings_df):
    """Validate extracted embeddings."""
    logger.info("Validating embeddings...")

    # Check dimensions
    embedding_cols = [f'A{i:02d}' for i in range(64)]
    missing_cols = [col for col in embedding_cols if col not in embeddings_df.columns]
    if missing_cols:
        logger.error(f"Missing embedding columns: {missing_cols}")
        return False

    # Check for NaN values
    n_samples = len(embeddings_df)
    nan_counts = embeddings_df[embedding_cols].isna().sum(axis=1)
    n_complete = (nan_counts == 0).sum()
    n_partial = ((nan_counts > 0) & (nan_counts < 64)).sum()
    n_missing = (nan_counts == 64).sum()

    logger.info(f"Embedding completeness:")
    logger.info(f"  Complete (64/64 dims): {n_complete} ({100*n_complete/n_samples:.1f}%)")
    logger.info(f"  Partial (1-63 dims):   {n_partial} ({100*n_partial/n_samples:.1f}%)")
    logger.info(f"  Missing (0/64 dims):   {n_missing} ({100*n_missing/n_samples:.1f}%)")

    # Check value ranges
    embedding_values = embeddings_df[embedding_cols].values.flatten()
    embedding_values = embedding_values[~pd.isna(embedding_values)]

    if len(embedding_values) > 0:
        logger.info(f"Embedding value range: [{embedding_values.min():.4f}, {embedding_values.max():.4f}]")
        logger.info(f"Embedding value mean: {embedding_values.mean():.4f}")

    return True

def save_embeddings(embeddings_df, input_file):
    """Save embeddings to TSV with provenance header."""
    output_file = os.path.join(OUTPUT_DIR, f"alphaearth_embeddings_gee_pfam_{TIMESTAMP}.tsv")

    embedding_cols = [f'A{i:02d}' for i in range(64)]

    # Create provenance header
    provenance = [
        "# Provenance:",
        f"#   Script: {os.path.abspath(__file__)}",
        f"#   Input: {input_file}",
        f"#   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"#   GEE Collection: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
        f"#   Scale: {GEE_SCALE}m",
        f"#   Total samples: {len(embeddings_df)}",
        f"#   Samples with complete embeddings: {(embeddings_df[embedding_cols].notna().all(axis=1)).sum()}",
        f"#   Purpose: Full coverage AlphaEarth embeddings for GCCA analysis",
    ]

    with open(output_file, 'w') as f:
        f.write('\n'.join(provenance) + '\n')
        embeddings_df.to_csv(f, sep='\t', index=False)

    logger.info(f"Saved embeddings to: {output_file}")
    return output_file

# =============================================================================
# Main
# =============================================================================

def main():
    """Main execution function."""
    logger.info("=" * 70)
    logger.info("AlphaEarth Extraction from GEE+PFAM Master File")
    logger.info("=" * 70)
    logger.info(f"Input: {INPUT_FILE}")
    logger.info(f"Output dir: {OUTPUT_DIR}")

    # Load coordinates
    coords_df = load_coordinates(INPUT_FILE)

    # Initialize GEE
    initialize_gee()

    # Extract embeddings
    embeddings_df = extract_all_embeddings(coords_df)

    # Validate
    validate_embeddings(embeddings_df)

    # Save
    output_file = save_embeddings(embeddings_df, INPUT_FILE)

    logger.info("=" * 70)
    logger.info("Extraction complete!")
    logger.info(f"Output: {output_file}")
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("  Update GCCA script to use the new embeddings file:")
    logger.info(f"    ALPHAEARTH_FILE = '{output_file}'")
    logger.info("=" * 70)

    return output_file

if __name__ == "__main__":
    main()
