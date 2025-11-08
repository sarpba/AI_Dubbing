FROM continuumio/anaconda3:2024.02-1

SHELL ["/bin/bash", "-c"]

WORKDIR /workspace

# System-level dependencies that are shared across the conda environments
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        gcc \
        git \
        hunspell \
        hunspell-en-us \
        hunspell-hu \
        libhunspell-dev \
        libsndfile1 \
        mkvtoolnix \
        mkvtoolnix-gui \
        pkg-config \
        rubberband-cli \
        sox \
        wget \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project sources first so that pip installs can leverage them during build
COPY . /workspace

# Additional repos referenced by the environment setup docs
RUN git clone https://github.com/sarpba/VibeVoice.git /opt/VibeVoice

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Prevent interactive prompts from conda during build
RUN conda config --set always_yes yes --set changeps1 no \
    && conda update -n base -c defaults conda

# --- sync environment (primary app runtime) ---
RUN conda create -n sync python=3.10 \
    && conda install -n sync -c conda-forge montreal-forced-aligner -y \
    && conda run -n sync python -m pip install -r /workspace/requirements.txt \
    && conda run -n sync python -m pip install elevenlabs yt-dlp pyrubberband textgrid

# --- demucs environment ---
RUN conda create -n demucs python=3.10 \
    && conda install -n demucs -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1 -y \
    && conda run -n demucs python -m pip install demucs numpy

# --- f5-tts environment ---
# CUDA-enabled PyTorch wheels by default; override if you need a different CUDA build.
ARG F5_TTS_TORCH_ARGS="torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124"
RUN conda create -n f5-tts python=3.11 \
    && conda run -n f5-tts python -m pip install --upgrade pip \
    && conda run -n f5-tts python -m pip install ${F5_TTS_TORCH_ARGS} \
    && conda run -n f5-tts python -m pip install git+https://github.com/SWivid/F5-TTS.git \
    && conda run -n f5-tts python -m pip install num2words Levenshtein nltk phonemizer hunspell

# --- nemo environment ---
# Install PyTorch with CUDA 12.1 wheels by default; override to match your stack if needed.
ARG NEMO_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
RUN conda create -n nemo python=3.10 \
    && conda run -n nemo python -m pip install --upgrade pip \
    && conda run -n nemo python -m pip install torch torchvision torchaudio --index-url ${NEMO_TORCH_INDEX_URL} \
    && conda run -n nemo python -m pip install nemo_toolkit[asr] librosa soundfile megatron megatron-core cuda-python \
    && conda install -n nemo -c conda-forge libstdcxx-ng -y

# --- vibevoice environment ---
RUN conda create -n vibevoice python=3.9 \
    && conda run -n vibevoice python -m pip install -e /opt/VibeVoice \
    && conda run -n vibevoice python -m pip install --upgrade pip \
    && conda run -n vibevoice python -m pip install openai-whisper transformers accelerate Levenshtein num2words numpy phonemizer hunspell

# --- whisperx environment ---
# CUDA-enabled wheels by default; override for different CUDA versions if necessary.
ARG WHISPERX_TORCH_ARGS="torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128"
RUN conda create -n whisperx python=3.10 \
    && conda install -n whisperx -c conda-forge ffmpeg sox libsndfile rust git pkg-config -y \
    && conda run -n whisperx python -m pip install --upgrade pip \
    && conda run -n whisperx python -m pip install ${WHISPERX_TORCH_ARGS} \
    && conda run -n whisperx python -m pip install "triton>=3.3.0" "nvidia-cudnn-cu12>=9.1.0" whisperx

RUN conda clean -afy

EXPOSE 2018

RUN chmod +x docker/entrypoint.sh

ENTRYPOINT ["./docker/entrypoint.sh"]
