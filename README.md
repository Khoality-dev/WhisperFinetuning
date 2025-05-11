# WhisperFinetuning

A minimal setup to fine‑tune OpenAI Whisper on your custom audio dataset.

## Requirements

* Python 3.9+
* ffmpeg (for audio conversion)

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Khoality-dev/WhisperFinetuning.git
   cd WhisperFinetuning
   ```
2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Dataset Structure

Organize your `dataset/` directory as follows:

```
dataset/
├── train.tsv      # audio_path <TAB> transcript
├── val.tsv        # audio_path <TAB> transcript
└── audio_files/   # .wav files
    ├── audio1.wav
    ├── audio2.wav
    └── ...
```

Optionally, generate data with:

```bash
python data_generation_tool.py
```

## Training

```bash
python train.py
```

## &#x20;

## License

MIT
