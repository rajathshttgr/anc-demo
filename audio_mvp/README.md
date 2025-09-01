# Smart Audio Enhancement & ANC (Python MVP)

## Features
- Feedforward ANC with LMS / NLMS
- Classical speech enhancement: Spectral Subtraction, Wiener Filter
- Offline pipeline with plots
- Real-time demo with PyAudio (two-channel input recommended)

## Install
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## Offline Demo (Jupyter)
1. Open `offline_demo.ipynb`
2. Set `primary_path` to a WAV file (speech+noise). Optionally set a `reference_path` (noise mic).
3. Run cells to view waveforms/spectrograms and export `demo_anc.wav` and `demo_enh.wav`.

## Real-time Demo
```bash
python -c "from smart_audio.pipeline_realtime import run_realtime; run_realtime()"
```
- For best results, provide two inputs (primary speech+noise, reference noise):
  - Adjust device indices in `run_realtime(...)` call.
- Press Ctrl+C to stop.

## Notes
- MP3 support: convert to WAV (e.g., `ffmpeg -i input.mp3 input.wav`) before using.
- Tune `filter_len`, `mu`, and optional `ref_delay` in `offline_process(...)` for stability.
- For presentations, use the notebook to plot before/after waveforms and spectrograms.
