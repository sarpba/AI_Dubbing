# AI Dubbing Application - New Backend Setup

This document outlines the necessary system packages and Python dependencies for running the new FastAPI-based backend (`main_app_new.py`) of the AI Dubbing application.

## System Package Dependencies

Before installing Python packages, ensure the following system tools are installed:

1.  **Anaconda or Miniconda**:
    *   The application uses `conda` to manage environments and run certain scripts (e.g., the TTS generation script in the `sync` environment).
    *   Installation instructions can be found on the [Anaconda Distribution page](https://www.anaconda.com/products/distribution) or [Miniconda page](https://docs.conda.io/en/latest/miniconda.html).
    *   After installation, ensure `conda` is initialized for your shell.

2.  **Git**:
    *   While `main_app_new.py` itself doesn't directly clone repositories, some underlying scripts or setup procedures might rely on Git. It's good practice to have it installed.
    *   On Debian/Ubuntu: `sudo apt update && sudo apt install git`
    *   On Fedora: `sudo dnf install git`
    *   On macOS (with Homebrew): `brew install git`

3.  **FFmpeg**:
    *   Many audio and video processing scripts often rely on FFmpeg for handling media files. It's highly recommended to have FFmpeg installed and available in your system's PATH.
    *   On Debian/Ubuntu: `sudo apt update && sudo apt install ffmpeg`
    *   On Fedora: `sudo dnf install ffmpeg`
    *   On macOS (with Homebrew): `brew install ffmpeg`

## Python Environment and Dependencies

1.  **Create and Activate Conda Environment (Recommended)**:
    It's recommended to run the backend in a dedicated Conda environment. If you already have the `sync` environment (as mentioned for the TTS script), you can use that or create a new one.
    ```bash
    # Example: Create a new environment named 'ai_dubbing_backend' with Python 3.9
    conda create -n ai_dubbing_backend python=3.9
    conda activate ai_dubbing_backend
    ```

2.  **Install Python Packages**:
    Install the necessary Python packages using the `requirements_new.txt` file:
    ```bash
    pip install -r requirements_new.txt
    ```
    This will install:
    *   `fastapi`: For the web framework.
    *   `uvicorn[standard]`: For the ASGI server to run FastAPI.
    *   `python-multipart`: For handling file uploads (e.g., MKV files).

3.  **Dependencies for Scripts in `scripts/` directory**:
    The `main_app_new.py` backend orchestrates various scripts located in the `scripts/` directory. These scripts have their own dependencies which are not explicitly listed in `requirements_new.txt` as they might be complex or managed within specific Conda environments (like the `sync` environment for `f5_tts_infer_API.py`).

    Ensure that the Conda environment(s) used by these scripts (e.g., `sync`) have all their necessary packages installed. Refer to the original project's `requirements.txt` or specific setup instructions for those scripts if issues arise. The key dependencies for those scripts likely include:
    *   `torch`, `torchaudio`
    *   `transformers`
    *   `librosa`
    *   `soundfile`
    *   `numpy`
    *   `pandas`
    *   `deepl`
    *   `whisperx` or `openai-whisper`
    *   Specific TTS libraries for F5-TTS.

    If you encounter `ModuleNotFoundError` when a script is run, you'll need to install the missing package into the appropriate Python environment (either the main backend environment or the specific Conda environment like `sync`).

## Running the Backend Application

Once all dependencies are installed:

1.  Navigate to the project directory where `main_app_new.py` is located.
2.  Ensure your Conda environment (e.g., `ai_dubbing_backend` or the one containing FastAPI) is activated.
3.  Run the FastAPI application using Uvicorn:
    ```bash
    uvicorn main_app_new:app --reload --port 8001
    ```
    *   `--reload`: Enables auto-reloading when code changes (useful for development).
    *   `--port 8001`: Specifies the port to run on (matching the frontend's `API_BASE_URL`).

The backend API will then be accessible at `http://localhost:8001`. The frontend (`frontend/index.html`) can then be opened in a browser to interact with the application.
