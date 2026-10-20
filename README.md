# Adaptive Signal Processing Based Noise Reduction System

Single-input denoiser plus a two-channel adaptive noise canceller (ANC) testbed,
written as plain NumPy so every step is explainable line by line.

## Pipeline

```
WAV -> mono, DC removal, peak normalise
    -> [1] adaptive notch (2-tap LMS, 50/100 Hz cascade, only if a tone is present)
    -> [2] adaptive line enhancer (optional, LMS on a delayed copy)
    -> [3] STFT 1024 / hop 256 / sqrt-Hann
             VAD (energy) -> noise PSD tracker (exponential, silence-gated)
             -> decision-directed Wiener gain (adaptive per bin, floored)
    -> ISTFT overlap-add -> clean WAV + metrics
```

The two-channel ANC path is the adaptive-filter showcase: a 32-tap LMS / NLMS /
RLS filter learns the acoustic path from a noise reference and subtracts it from
the noisy primary signal.

## Quick start

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python -m pytest -q                    # 22 tests
.venv/bin/python benchmarks/run_benchmark.py     # figures + table in results/
```

## Layout

```
dsp/
  framing.py     STFT/ISTFT, sqrt-Hann, overlap-add (exact reconstruction)
  lms.py         LMS, NLMS, RLS (MSE curves returned for plots)
  notch.py       adaptive notch, harmonic cascade, adaptive line enhancer
  noise_est.py   VAD, spectral flatness, adaptive noise-PSD tracker
  spectral.py    decision-directed Wiener gain, spectral subtraction baseline
  pipeline.py    config + end-to-end denoise(), keeps every intermediate signal
  metrics.py     SNR, SI-SDR, ERLE, segmental SNR
  synth.py       speech-like signal and noise generators for reproducible tests
benchmarks/
  run_benchmark.py   grid over noise types and SNRs, ANC convergence, figures
tests/                unit tests for every module
```

## Results (phase 1, 12.6 s speech, SI-SDR dB averaged over 0/5/10 dB input)

| noise  | noisy | spectral subtraction | adaptive Wiener | gain |
|--------|-------|----------------------|------------------|------|
| white  | 5.01  | 11.50 | 11.94 | +6.93 |
| pink   | 5.06  | 10.65 | 10.80 | +5.74 |
| brown  | 5.38  | 13.00 | 13.00 | +7.62 |
| hum    | 4.97  | 16.80 | 15.95 | +10.98 |
| fan    | 5.02  | 10.23 | 10.01 | +5.00 |
| babble | 4.94  | 9.60  | 9.11  | +4.17 |

Two-channel ANC with speech present (6 s, fan noise, 32 taps):

| filter           | noisy SI-SDR | clean SI-SDR | ERLE | blocks to 20 dB |
|------------------|--------------|--------------|------|-----------------|
| LMS mu=0.002     | -12.21 | 2.74  | 13.13 | 11 |
| NLMS mu=0.5      | -12.21 | -4.20 | 14.05 | 5  |
| NLMS mu=0.1      | -12.21 | 2.59  | 12.81 | 8  |
| RLS lambda=0.995 | -12.21 | 7.93  | 12.02 | 0  |

NLMS with a large step size converges fastest but has the largest misadjustment;
RLS converges inside one block. This tradeoff is the core discussion of the
report and is illustrated in `results/figures/`.

## Status

- Phase 1: DSP core, tests, benchmark - done
- Phase 2: FastAPI backend + web UI - next
- Phase 3: tuning, report assets, viva demo script - planned
