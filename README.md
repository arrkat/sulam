# Sulam — Audio Key Detection Tool

**Sulam** is a Python-based application designed to estimate the musical key of audio files accurately and efficiently. Leveraging chroma feature extraction and custom heuristics, Sulam provides musicians, DJs, and producers with quick insights into the tonal center of their tracks.

---

## Features

- **Accurate Key Detection:** Estimates major and minor keys (natural, harmonic, melodic).
- **Support for Multiple Audio Formats:** Works with MP3, WAV, FLAC, and more.
- **Lightweight Processing:** Uses bandpass filtering and chroma-based analysis for fast results.
- **Interactive Local UI:** Simple Gradio interface with local file picker—no uploads or cloud processing.
- **Customizable Algorithm:** Weighted scoring on tonic, third, and fifth intervals to refine detection.

---

### Requirements

- Python 3.11+
- [Homebrew](https://brew.sh/) (optional, for macOS users)
- `tkinter` for local file picker support (usually pre-installed on Windows/macOS)
  - On Linux, you may need to install it via your package manager, e.g.:
    ```bash
    sudo apt install python3-tk
    ```

