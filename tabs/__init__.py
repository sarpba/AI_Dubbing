# tabs/__init__.py
from .project_creation import upload_and_extract_audio
from .transcription import transcribe_audio_whisperx
from .speech_removal import separate_audio
from .audio_splitting import split_audio
#from .chunk_verification import verify_chunks_whisperx, compare_transcripts_whisperx
from .verify_chunks import verify_chunks_whisperx
from .compare_transcripts import compare_transcripts_whisperx
from .translate import translate_chunks
from .merge_chunks import merge_chunks 
from .integrate_audio import integrate_audio
from .adjust_audio import adjust_audio
from .tts_generation import tts_generation
from .huntextnormalizer import HungarianTextNormalizer
