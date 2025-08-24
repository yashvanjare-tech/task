import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os

# Load pre-trained model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def load_audio(file_path):
    speech, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        speech = resampler(speech)
    return speech.squeeze(), 16000

def transcribe_audio(file_path):
    speech, rate = load_audio(file_path)
    input_values = processor(speech, return_tensors="pt", sampling_rate=rate).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.decode(predicted_ids[0])

if __name__ == "__main__":
    audio_path = input("Enter the path to the .wav audio file: ").strip()
    if not os.path.exists(audio_path):
        print("File not found. Please check the path and try again.")
    else:
        text = transcribe_audio(audio_path)
        print("Transcription:\n", text)
