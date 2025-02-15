# 1) Alapképként Ubuntu 22.04
FROM ubuntu:22.04

# APT interaktív kérdések elkerülése (opcionális)
ENV DEBIAN_FRONTEND=noninteractive

# 2) Alaprendszer frissítése, szükséges csomagok telepítése
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    git \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 3) Anaconda letöltése és telepítése (2022.10 példa)
RUN wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh -O /tmp/anaconda.sh && \
    bash /tmp/anaconda.sh -b -p /opt/conda && \
    rm /tmp/anaconda.sh

# 4) Conda parancsok elérhetősége az útvonalban
ENV PATH="/opt/conda/bin:${PATH}"

# 5) A szükséges conda-környezetek és csomagok létrehozása
RUN conda create -y -n f5-tts python=3.10 pip && \
    /bin/bash -c "source activate f5-tts && \
        conda install -y git && \
        pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 && \
        pip install git+https://github.com/SWivid/F5-TTS.git && \
        pip install num2words && \
        conda deactivate" && \
    conda create -y -n sync python=3.10 pip && \
    /bin/bash -c "source activate sync && \
        pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 && \
        conda install -y git && \
        pip install git+https://github.com/m-bain/whisperx.git && \
        conda install -y cudatoolkit=11.8 cudnn=8.9* && \
        conda deactivate"

# 6) A projekt fájlok letöltése, telepítése
#    (Példában a /home/sarpba könyvtárban dolgozunk)
RUN mkdir -p /home/sarpba && cd /home/sarpba && \
    git clone https://github.com/sarpba/AI_Dubbing.git && \
    cd AI_Dubbing && \
    /bin/bash -c "source activate sync && pip install -r requirements.txt" && \
    mkdir -p /home/sarpba/AI_Dubbing/TTS/hun_v5 && \
    cd /home/sarpba/AI_Dubbing/TTS/hun_v5 && \
    wget --header="Authorization: Bearer WRITE_YOUR_ACCES_TOKEN_HERE!!!!!!!!!" \
         https://huggingface.co/sarpba/F5-TTS-Hun/resolve/main/hun_v5/model_250000_quant.pt \
         https://huggingface.co/sarpba/F5-TTS-Hun/resolve/main/hun_v5/vocab.txt

# 7) Alapértelmezett munkakönyvtár
WORKDIR /home/sarpba/AI_Dubbing

ENTRYPOINT ["/home/sarpba/AI_Dubbing/docker/entrypoint.sh"]

# 8) Konténer induláskor futtatandó parancs
#    - A conda run lekezeli a környezetet,
#      így nincs szükség "source activate" hívásra.
CMD ["conda", "run", "-n", "sync", "python", "main_app.py"]
