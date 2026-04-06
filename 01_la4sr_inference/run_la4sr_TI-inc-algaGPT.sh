#!/usr/bin/env bash
set -euo pipefail
###############################################################################
# run_la4sr_TI-inc.sh — LA4SR pipeline: model inference (FASTA→TSV) + metrics
###############################################################################

# ---------------------- Configuration ----------------------
SCRIPT_DIR="$(pwd)"
INFER_SCRIPT="$SCRIPT_DIR/infer_TI-inc-algaGPT.py"   # <— new script
METRICS_SCRIPT="$SCRIPT_DIR/llm-metrics-two-files.py"
SIF="$SCRIPT_DIR/la4sr_sp2.sif"

# cache for HF tokenizers & models
tcache="$SCRIPT_DIR/cache"
mkdir -p "$tcache"

# ---------------------- Usage ----------------------
if [[ $# -ne 3 ]]; then
  cat <<EOF
Usage: $(basename "$0") <model_name|resume> <algal_fasta> <bacterial_fasta>

If you pass the literal word   resume   the script loads ckpt.pt + meta.pkl
from the current directory. Otherwise the value is forwarded to --init_from
(e.g. GreenGenomicsLab/LA4SR-gpt-neo125-ALMGA-FL).

EOF
  exit 1
fi

MODEL_NAME="$1"          # "resume"  OR  HF repo / local path
algal_fasta="$2"
bact_fasta="$3"

prefix="$(basename "${algal_fasta%.*}")_vs_$(basename "${bact_fasta%.*}")"
mkdir -p results
alg_out="results/${prefix}_algal.tsv"
bac_out="results/${prefix}_bacterial.tsv"
alg_out_tagged="results/${prefix}_algal_tagged.tsv"
bac_out_tagged="results/${prefix}_bacterial_tagged.tsv"
report="results/${prefix}_report.txt"
miscl="results/${prefix}_misclassified.txt"

# ---------------------- Inference ----------------------

run_infer () {
  local fasta=$1 out=$2
  echo -e "\n→ Inferring $(basename "$fasta")..."

  # Build common args
  PY_ARGS=( --init_from "$MODEL_NAME" )
  [[ "$MODEL_NAME" == "resume" ]] && PY_ARGS+=( --out_dir /workdir )

  singularity exec --nv \
    -B "$fasta:/input.fasta" \
    -B "$(pwd):/workdir" \
    -B "$tcache:$tcache" \
    --env TRANSFORMERS_CACHE="$tcache" \
    "$SIF" \
    bash -c 'cd /workdir && \
      python3 infer_TI-inc-algaGPT.py '"${PY_ARGS[*]}"' /input.fasta -o "'"$out"'"'
}

#run_infer () {
  #local fasta=$1 out=$2
  #echo -e "\n→ Inferring $(basename "$fasta")..."

  #PY_ARGS=( --init_from "$MODEL_NAME" )
  #[[ "$MODEL_NAME" == "resume" ]] && PY_ARGS+=( --out_dir /workdir )

 # singularity exec --nv \
   # -B "$fasta:/input.fasta" \
   # -B "$(pwd):/workdir" \          # <— whole project goes in
   # -B "$tcache:$tcache" \
   # --env TRANSFORMERS_CACHE="$tcache" \
   # "$SIF" \
   # bash -c "cd /workdir && \
  #           python3 "$INFER_SCRIPT" \
 #              \"${PY_ARGS[@]}\" /input.fasta -o \"$out\""
#}

#run_infer () {
 # local fasta=$1 out=$2
  #echo -e "\n→ Inferring $(basename "$fasta")..."

  # build python arg list: --init_from ... [--out_dir PWD]
  #PY_ARGS=( --init_from "$MODEL_NAME" )
  #[[ "$MODEL_NAME" == "resume" ]] && PY_ARGS+=( --out_dir "$SCRIPT_DIR" )

  #if [[ -f "$SIF" ]]; then
   # singularity exec --nv \
    #  -B "$fasta:/input.fasta" \
     # -B "$INFER_SCRIPT:/infer.py" \
      #-B "$(pwd):/workdir" \
      #-B "$tcache:$tcache" \
      #--env TRANSFORMERS_CACHE="$tcache" \
      #"$SIF" \
      #python3 /infer.py "${PY_ARGS[@]}" /input.fasta \
       # -o "/workdir/$out"
  #else
   # TRANSFORMERS_CACHE="$tcache" \
    #python3 "$INFER_SCRIPT" "${PY_ARGS[@]}" "$fasta" \
     # -o "$out"
  #fi

  #echo "   ✔ Wrote $out"
#}

run_infer "$algal_fasta" "$alg_out"
run_infer "$bact_fasta"  "$bac_out"

# ---------------------- Post-process Tags ----------------------
convert_tags () {
  local infile=$1 outfile=$2
  echo -e "\n→ Converting 'algae'→@ and 'conta'→! in $(basename "$infile")..."
  sed -E 's/algae/@/g; s/conta/!/g' "$infile" > "$outfile"
  echo "   ✔ Wrote $outfile"
}

convert_tags "$alg_out" "$alg_out_tagged"
convert_tags "$bac_out" "$bac_out_tagged"

# ---------------------- Metrics ----------------------
echo -e "\n→ Generating metrics report..."
singularity exec \
  -B "$METRICS_SCRIPT:/metrics.py" \
  -B "$(pwd):/workdir" \
  "$SIF" \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && \
           conda activate la4sr && cd /workdir && \
           python3 /metrics.py \
             \"$alg_out_tagged\" \"$bac_out_tagged\" \
             -o \"$report\" \
             -m \"$miscl\" \
             -v \
             -p \"results/$prefix\""

# ---------------------- Finished ----------------------
echo -e '\n🎉 Done! Results in ./results/'
echo "  Algal TSV:      $alg_out_tagged"
echo "  Bact TSV:       $bac_out_tagged"
echo "  Report:         $report"
echo "  Misclassified:  $miscl"
echo "  Plots:          results/${prefix}_*.png"

