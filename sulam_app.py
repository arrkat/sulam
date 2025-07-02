import gradio as gr
import librosa
import numpy as np
import scipy.signal
from collections import namedtuple
import tkinter as tk
from tkinter import filedialog

PITCHES = ['C', 'C#', 'D', 'D#', 'E', 'F',
           'F#', 'G', 'G#', 'A', 'A#', 'B']

SCALE_TEMPLATES = {
    "Major": np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]),
    "Minor (Natural)": np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]),
    "Minor (Harmonic)": np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1]),
    "Minor (Melodic)": np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1])
}


KeyMatch = namedtuple("KeyMatch", ["name", "mode", "score"])

# Helpers
def choose_local_file():
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.flac")])
    return filepath

def select_file_and_update_textbox():
    filepath = choose_local_file()
    return filepath  # this will update the Textbox
    
def bandpass_filter(signal, sample_rate, low_freq=80, high_freq=1000, order=5):
    sos = scipy.signal.butter(order, [low_freq, high_freq], btype='bandpass', fs=sample_rate, output='sos')
    return scipy.signal.sosfilt(sos, signal)

def compute_average_chroma(signal, sample_rate):
    """
    Compute a weighted and normalized average chroma vector from an audio signal.

    Parameters:
        signal (np.ndarray): Audio time series.
        sample_rate (int): Sampling rate of the audio signal.

    Returns:
        np.ndarray: Normalized average chroma vector (length 12).
    """
    # Extract chroma features using the CENS method, starting from note C2
    chroma = librosa.feature.chroma_cens(y=signal, sr=sample_rate, fmin=librosa.note_to_hz('C2'))

    # Weight frames by their max chroma value to emphasize strong harmonic content
    frame_weights = np.max(chroma, axis=0)

    # Compute weighted average across time frames
    avg_chroma = np.average(chroma, axis=1, weights=frame_weights)

    # Ensure all values are non-negative
    avg_chroma = np.clip(avg_chroma, a_min=0, a_max=None)

    # Normalize the chroma vector using L1 norm for consistency
    avg_chroma /= np.linalg.norm(avg_chroma, ord=1)

    return avg_chroma


def estimate_key(avg_chroma, root_weight=1.3, third_weight=1.1, fifth_weight=1.1):
    """
    Estimate the musical key from an average chroma vector.

    Parameters:
        avg_chroma (np.ndarray): Average chroma vector (length 12).
        root_weight (float): Weight to boost the tonic/root note.
        third_weight (float): Weight to boost the 3rd interval (major or minor).
        fifth_weight (float): Weight to boost the perfect 5th interval.

    Returns:
        best (KeyMatch): Best matching key (pitch, mode, and score).
        sorted_scores (list of KeyMatch): All key matches sorted by score descending.
    """

    def score(chroma_vector, scale_template, root_index, mode):
        # Rotate the scale template so the root is at index 0
        rolled_template = np.roll(scale_template, root_index)

        # Initialize equal weights for all 12 pitch classes
        weights = np.ones(12)

        # Boost the root (tonic) weight
        weights[root_index] = root_weight

        # Determine the interval for the 3rd degree (major or minor)
        third_interval = 4 if mode == "Major" else 3
        weights[(root_index + third_interval) % 12] = third_weight

        # Boost the perfect 5th interval
        weights[(root_index + 7) % 12] = fifth_weight

        # Compute weighted dot product between chroma and scale template
        return np.dot(chroma_vector * weights, rolled_template)

    # Compute scores for all keys and modes
    scores = [
        KeyMatch(PITCHES[i], mode, score(avg_chroma, template, i, mode))
        for mode, template in SCALE_TEMPLATES.items()
        for i in range(len(PITCHES))
    ]

    # Select best key match
    best = max(scores, key=lambda k: k.score)

    # Sort all scores descending for detailed output or debugging
    sorted_scores = sorted(scores, key=lambda k: k.score, reverse=True)

    return best, sorted_scores
    
    
def detect_key_from_audio(filepath):
    """
    Detect the musical key of an audio file.

    Parameters:
        filepath (str): Path to the audio file.

    Returns:
        str: Formatted string with the best estimated key and top 3 candidate keys.
             Returns an error message if no filepath is provided.
    """
    if not filepath:
        return "No file provided"

    # Load audio with a standard sampling rate
    signal, sr = librosa.load(filepath, sr=22050)

    # Apply bandpass filter to focus on relevant frequency range
    filtered_signal = bandpass_filter(signal, sr)

    # Compute the average chroma vector from the filtered signal
    avg_chroma = compute_average_chroma(filtered_signal, sr)

    # Estimate the key using the chroma features
    best_match, all_matches = estimate_key(avg_chroma)

    # Prepare a formatted string showing the top 3 key candidates
    top_candidates = "\n".join(
        f"{match.name} {match.mode}: {match.score:.4f}" for match in all_matches[:3]
    )

    return (
        f"🎼 Estimated Key: {best_match.name} {best_match.mode}\n\n"
        f"Top 3 Candidates:\n{top_candidates}"
    )


# === Gradio Blocks UI ===
with gr.Blocks() as demo:
    gr.Markdown("## Key Detection (Local File Upload)")

    with gr.Row():
        file_input = gr.File(label="Select Audio File", file_types=[".mp3", ".wav", ".flac"], type="filepath")
        detect_btn = gr.Button("Detect Key")

    result_box = gr.Textbox(label="Detected Key")

    detect_btn.click(fn=detect_key_from_audio, inputs=file_input, outputs=result_box)

demo.launch()
