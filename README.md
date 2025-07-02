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

## Installation

### Prerequisites

- Python 3.10 or newer
- [Homebrew](https://brew.sh/) (optional, for macOS users)

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/sulam.git
   cd sulam
Create a virtual environment (recommended)

bash
Copy
Edit
python3.11 -m venv sulam-env
source sulam-env/bin/activate   # macOS/Linux
sulam-env\Scripts\activate.bat  # Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt