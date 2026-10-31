# Fingerprint Integrity Lab
Local FastAPI website for the ICTCon 2026 seam-carving fingerprint-forensics demonstration.

## Run locally
```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```
Open http://127.0.0.1:8000.

## Included
- Research overview and reported grouped-CV benchmark results.
- Single-image upload and 5%, 10%, 20% seam-carving simulation.
- Single-image 320-D LBP + Gradient feature extraction and ratio-specific SVM inference.
- Optional NBIS mindtct + Bozorth3 pair comparison.

## Models
Place `svm_5.joblib`, `svm_10.joblib`, and `svm_20.joblib` in `artifacts/`. Model binaries are not included. `train_and_export_models.py` is the modified exporter; check its expected NPZ filenames against your actual feature files before running.

## Feature parity
The live extractor follows the supplied STEP 2 implementation: grayscale OpenCV decoding; `local_binary_pattern(gray, 8, 1, method='default')`; 256-bin density histogram over [0,256]; OpenCV Sobel CV_64F, ksize=3; gradient magnitude 64-bin density histogram over [0,512]; concatenate LBP then gradient. No additional L2 normalization is applied. Verify vectors against the training NPZ before interpreting live predictions scientifically.

## Limitations
The seam-carving endpoint is a reference backward-energy implementation followed by bilinear restoration. Verify equivalence with the exact library and parameters used to generate the experimental data. Live AFIS requires NBIS executables on PATH. Paper-reported aggregate metrics are not live predictions.
