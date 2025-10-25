Esetleges korábbi telepítés törlése

```bash
conda deactivate
conda env remove -n f5-tts
```

Anakonda (nem minikonda) Környezet építése

```bash
conda create -n f5-tts python=3.11 -y && conda activate f5-tts
conda install git
#pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
# Install pytorch with your CUDA version, e.g.
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/SWivid/F5-TTS.git
pip install num2words Levenshtein nltk

```
A magyar szövegnormalizáló függőségei:

Telepítsd a kötelező csomagot, illetve az opcionális fonemizálást és hunspell-támogatást:

```bash
conda install -c conda-forge num2words phonemizer
```

Ha a phonemizer gondot jelez, pipről frissítheted:

```bash
pip install --upgrade phonemizer
```
huspell telepítése
```bash
sudo apt-get update
sudo apt-get install -y libhunspell-dev hunspell
# (optional dictionaries)
sudo apt-get install -y hunspell-en-us hunspell-hu
# back to your env
conda activate f5-tts
pip install --no-cache-dir hunspell
```


Az ezutáni rész elvileg nem szükséges.

A Hunspell szótárakhoz másold a .dic/.aff fájlokat a normalisers/hun/hunspell/ mappába, vagy állítsd be a HUNSPELL_DICT_PATH változót:

```bash
mkdir -p normalisers/hun/hunspell
export HUNSPELL_DICT_PATH=/home/sarpba/AI_Dubbing/normalisers/hun/hunspell
```

(Opcionális) Ha a rendszereden nincs magyar hunspell szótár, tölts le egyet és másold ebbe a mappába:

```bash
wget https://cgit.freedesktop.org/libreoffice/dictionaries/plain/hu_HU/hu_HU.dic
wget https://cgit.freedesktop.org/libreoffice/dictionaries/plain/hu_HU/hu_HU.aff
mv hu_HU.* normalisers/hun/hunspell/
```