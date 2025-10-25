Esetleges korábbi telepítés törlése

```bash
conda deactivate
conda env remove -n vibevoice
```

Anakonda (nem minikonda) Környezet építése

```bash
git clone https://github.com/sarpba/VibeVoice.git
cd VibeVoice
conda create --name vibevoice python=3.9 -y
conda activate vibevoice
pip install -e .
pip install openai-whisper
pip install transformers accelerate
pip install Levenshtein num2words
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
conda activate vibevioce
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