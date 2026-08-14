#!/bin/bash
# Builds the URL training corpus for train.py into model/data/.
#
# Sources:
# - Common Crawl URL index (https://commoncrawl.org/): ~600 index
#   blocks sampled uniformly across the SURT-sorted domain space, via
#   the cluster.idx block map, so the corpus covers many hosts instead
#   of one alphabetical neighborhood
# - https://github.com/ada-url/url-dataset: ~100K URLs from popular
#   sites (upweighted during training - they're what people shorten)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/ccblocks
cd data

CRAWL="CC-MAIN-2025-26"
BASE="https://data.commoncrawl.org/cc-index/collections/$CRAWL/indexes"

if [ ! -s cluster.idx ]; then
  echo "Downloading cluster.idx (~110MB)..."
  curl -s -o cluster.idx "$BASE/cluster.idx"
fi

# Sample every 1600th block pointer: shard file, byte offset, length
awk 'NR % 1600 == 7 { split($0, a, "\t"); print a[2], a[3], a[4] }' cluster.idx > blocks.txt
echo "Fetching $(wc -l < blocks.txt) index blocks..."

fetch_block() {
  local f="ccblocks/$1.$2.gz"
  [ -s "$f" ] && return 0
  local end=$(( $2 + $3 - 1 ))
  curl -s --max-time 60 --retry 2 -r "$2-$end" -o "$f" "$BASE/$1" || rm -f "$f"
}
export -f fetch_block
export BASE
xargs -a blocks.txt -L1 -P6 bash -c 'fetch_block "$@"' _

if [ ! -d url-dataset ]; then
  git clone --depth 1 https://github.com/ada-url/url-dataset
fi

python3 ../extract-urls.py
echo "Done. Corpus in data/cc-urls.txt + data/url-dataset/out.txt"
