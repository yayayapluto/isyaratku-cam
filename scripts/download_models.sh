#!/usr/bin/env bash
# Unduh model ke models/ (TIDAK di-commit).
# Sumber:
#   - HF Syizuril/bisindo-sign-language (publik, tanpa token, lisensi TIDAK tertulis)
#   - HF rhasspy/piper-voices (id/id_ID/news_tts/medium, lisensi "See URL" -> belum terverifikasi)
#   - Google MediaPipe hand_landmarker (Apache-2.0)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT/models/bisindo_alphabet"
VOICE_DIR="$ROOT/models/voices"
HAND_DIR="$ROOT/models/hand"
HF="https://huggingface.co"
MP="https://storage.googleapis.com/mediapipe-models"

mkdir -p "$MODEL_DIR" "$VOICE_DIR" "$HAND_DIR"

if ls "$MODEL_DIR"/*.pth >/dev/null 2>&1; then
    echo "model A-Z sudah ada, lewati"
else
    echo "unduh bobot A-Z..."
    curl -fL --retry 3 -o "$MODEL_DIR/efficientnet_bisindo_sign_language.pth" \
        "$HF/Syizuril/bisindo-sign-language/resolve/main/efficientnet_bisindo_sign_language.pth"
fi

if ls "$VOICE_DIR"/id_ID-news_tts-medium.onnx >/dev/null 2>&1; then
    echo "voice Piper sudah ada, lewati"
else
    echo "unduh voice Piper id_ID-news_tts-medium (~63MB)..."
    V="$HF/rhasspy/piper-voices/resolve/main/id/id_ID/news_tts/medium/id_ID-news_tts-medium"
    curl -fL --retry 3 -o "$VOICE_DIR/id_ID-news_tts-medium.onnx" "$V.onnx"
    curl -fL --retry 3 -o "$VOICE_DIR/id_ID-news_tts-medium.onnx.json" "$V.onnx.json"
fi

if [ -f "$HAND_DIR/hand_landmarker.task" ]; then
    echo "hand landmarker sudah ada, lewati"
else
    echo "unduh MediaPipe hand landmarker (~7.5MB)..."
    curl -fL --retry 3 -o "$HAND_DIR/hand_landmarker.task" \
        "$MP/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
fi

echo "selesai"
echo "verifikasi: python scripts/check_env.py"
