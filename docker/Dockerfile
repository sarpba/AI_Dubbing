# Alap image
FROM ubuntu:22.04

# Használjuk a bash-t a RUN parancsokhoz
SHELL ["/bin/bash", "-c"]

# Fogadjuk el a Hugging Face token-t build argumentumként
ARG HF_TOKEN

# Telepítjük az alap csomagokat
RUN apt-get update && \
    apt-get install -y wget bzip2 ca-certificates git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Anaconda telepítése
RUN wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh -O /tmp/anaconda.sh && \
    bash /tmp/anaconda.sh -b -p /opt/conda && \
    rm /tmp/anaconda.sh

# Frissítjük az elérési utat, hogy az Anaconda eszközök elérhetőek legyenek
ENV PATH="/opt/conda/bin:$PATH"

# Létrehozzuk és konfiguráljuk a f5-tts környezetet
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda create -y -n f5-tts python=3.10 pip && \
    conda activate f5-tts && \
    conda install -y git && \
    pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 && \
    pip install git+https://github.com/SWivid/F5-TTS.git && \
    pip install num2words

# Létrehozzuk és konfiguráljuk a sync környezetet
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda create -y -n sync python=3.10 pip && \
    conda activate sync && \
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 && \
    conda install -y git && \
    pip install git+https://github.com/m-bain/whisperx.git && \
    conda install -y cudatoolkit=11.8 cudnn=8.9*

# Klónozzuk a projektet és telepítjük a függőségeket
RUN mkdir -p /home/sarpba && \
    cd /home/sarpba && \
    git clone https://github.com/sarpba/AI_Dubbing.git && \
    cd AI_Dubbing && \
    pip install -r requirements.txt

# Hozzuk létre a szükséges könyvtárat a modellek számára
RUN mkdir -p /home/sarpba/AI_Dubbing/TTS/hun_v5

# Letöltjük a modellt és a szólistát a megadott HF tokennel
RUN cd /home/sarpba/AI_Dubbing/TTS/hun_v5 && \
    wget --header="Authorization: Bearer ${HF_TOKEN}" \
         https://huggingface.co/sarpba/F5-TTS-Hun/resolve/main/hun_v5/model_250000_quant.pt && \
    wget --header="Authorization: Bearer ${HF_TOKEN}" \
         https://huggingface.co/sarpba/F5-TTS-Hun/resolve/main/hun_v5/vocab.txt

# Adjuk meg a futtatási jogot az entrypoint scriptnek
RUN chmod +x /home/sarpba/AI_Dubbing/docker/entrypoint.sh

# Az entrypoint beállítása: ez a script fog futni, amikor a konténer indul
ENTRYPOINT ["/home/sarpba/AI_Dubbing/docker/entrypoint.sh"]
