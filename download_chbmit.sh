#!/bin/bash
# Download CHB-MIT Scalp EEG Database from PhysioNet
# Dataset: https://physionet.org/content/chbmit/1.0.0/
# 
# This script downloads EEG recordings for 4 patients:
#   - chb01, chb02, chb03, chb05
#
# Total size: ~2-3 GB
# Expected time: 20-30 minutes (depending on connection)
#
# Usage:
#   chmod +x download_chbmit.sh
#   ./download_chbmit.sh

DATA_DIR="$HOME/chbmit_data"
mkdir -p $DATA_DIR

# Download specific patients (CHB01, CHB02, CHB03, CHB05)
BASE_URL="https://physionet.org/files/chbmit/1.0.0"

echo "=========================================="
echo "CHB-MIT Dataset Download"
echo "=========================================="
echo "Downloading to: $DATA_DIR"
echo ""

for PATIENT in chb01 chb02 chb03 chb05; do
    echo "Downloading $PATIENT..."
    mkdir -p $DATA_DIR/$PATIENT
    
    # Download the patient directory listing
    wget -q -O - $BASE_URL/$PATIENT/ | \
        grep -oP '(?<=href=")[^"]*\.edf(?=")' | \
        while read file; do
            echo "  Downloading $file..."
            wget -q -nc -P $DATA_DIR/$PATIENT $BASE_URL/$PATIENT/$file
        done
    
    # Download summary file
    echo "  Downloading ${PATIENT}-summary.txt..."
    wget -q -nc -P $DATA_DIR/$PATIENT $BASE_URL/$PATIENT/${PATIENT}-summary.txt
    
    echo "  $PATIENT complete."
    echo ""
done

echo "=========================================="
echo "Download complete!"
echo "Data saved to: $DATA_DIR"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Extract features:"
echo "     python pacan_week8.py --extract_features --patient chb01 --data_dir $DATA_DIR"
echo ""
echo "  2. Fit WC parameters:"
echo "     python pacan_week8.py --fit_wc --patient chb01"
echo ""
echo "  3. Build pairwise MEM:"
echo "     python pacan_week8.py --build_mem --patient chb01"
