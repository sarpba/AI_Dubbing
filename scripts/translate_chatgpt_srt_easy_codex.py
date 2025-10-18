import os
import argparse
import json
import re
import time
import glob
import sys
import base64
from pathlib import Path
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple
from bisect import bisect_left

import srt  # pip install srt
from openai import OpenAI, OpenAIError

for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from tools.debug_utils import add_debug_argument, configure_debug_mode

# --- KONFIGURÁCIÓ ÉS KULCSKEZELÉS -------------------------------------------------

def get_project_root() -> str:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    if os.path.basename(project_root) == 'scripts':
        return os.path.abspath(os.path.join(script_dir, '..'))
    return os.getcwd()


def load_config() -> Optional[Dict[str, Any]]:
    config_path = os.path.join(get_project_root(), 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("Konfigurációs fájl sikeresen betöltve.")
        return config
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        print(f"Hiba a konfigurációs fájl betöltése közben ({config_path}): {e}")
        return None


def get_keyholder_path() -> str:
    return os.path.join(get_project_root(), 'keyholder.json')


def save_api_key(api_key: str) -> None:
    path = get_keyholder_path()
    try:
        data: Dict[str, Any] = {}
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print("Figyelmeztetés: A keyholder.json sérült vagy üres, új fájl jön létre.")
                    data = {}
        encoded_key = base64.b64encode(api_key.encode('utf-8')).decode('utf-8')
        data['chatgpt_api_key'] = encoded_key
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"ChatGPT API kulcs sikeresen elmentve a(z) '{path}' fájlba.")
    except Exception as e:
        print(f"Hiba az API kulcs mentése közben: {e}")


def load_api_key() -> Optional[str]:
    path = get_keyholder_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        encoded_key = data.get('chatgpt_api_key')
        if not encoded_key:
            return None
        decoded_key = base64.b64decode(encoded_key.encode('utf-8')).decode('utf-8')
        return decoded_key
    except (json.JSONDecodeError, KeyError, base64.binascii.Error, Exception) as e:
        print(f"Hiba az API kulcs betöltése közben a(z) '{path}' fájlból: {e}")
        return None


# --- FÁJLKEZELÉS ÉS SEGÉDFÜGGVÉNYEK ----------------------------------------------

def find_srt_context_file(upload_dir: str, lang_code: str) -> Optional[str]:
    if not os.path.isdir(upload_dir):
        print(f"Figyelmeztetés: Az 'upload' könyvtár nem található: {upload_dir}. Folytatás SRT kontextus nélkül.")
        return None
    search_pattern = os.path.join(upload_dir, f'*{lang_code.lower()}.srt')
    print(f"  -> Kontextusfájl keresése a következő mintával: {search_pattern}")
    matching_files = glob.glob(search_pattern)
    if not matching_files:
        print(f"  -> Nem található a(z) '*{lang_code.lower()}.srt' mintának megfelelő SRT fájl a(z) '{upload_dir}' könyvtárban.")
        return None
    if len(matching_files) == 1:
        selected_file = matching_files[0]
        print(f"  -> SRT kontextusfájl megtalálva: {os.path.basename(selected_file)}")
        return selected_file
    print(f"  -> Több ({len(matching_files)}) lehetséges SRT fájl található. A legnagyobb kiválasztása...")
    try:
        largest_file = max(matching_files, key=os.path.getsize)
        print(f"  -> A legnagyobb fájl kiválasztva: {os.path.basename(largest_file)} ({os.path.getsize(largest_file) / 1024:.2f} KB)")
        return largest_file
    except Exception as e:
        print(f"Hiba a legnagyobb fájl kiválasztása közben: {e}. Az első találat használata: {os.path.basename(matching_files[0])}")
        return matching_files[0]


def find_json_file(directory: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        json_files = [f for f in os.listdir(directory) if f.lower().endswith('.json')]
    except FileNotFoundError:
        print(f"Hiba: A bemeneti könyvtár nem található: {directory}")
        return None, "directory_not_found"
    if not json_files:
        print(f"Hiba: Nem található JSON fájl a(z) '{directory}' könyvtárban.")
        return None, "no_json_found"
    if len(json_files) > 1:
        print(f"Hiba: Több JSON fájl található a(z) '{directory}' könyvtárban.")
        return None, "multiple_jsons_found"
    return json_files[0], None


def clean_srt_content(text: str) -> str:
    text = re.sub(r'\{[^\}]*\}', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    return text.strip()


def load_srt_file(filepath: Optional[str]) -> Tuple[List[Dict[str, Any]], str]:
    if not filepath or not os.path.exists(filepath):
        if filepath:
            print("  -> Figyelmeztetés: Az SRT fájl nem található a megadott útvonalon.")
        return [], ""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        subtitles = list(srt.parse(content))
        processed: List[Dict[str, Any]] = []
        for order, sub in enumerate(subtitles):
            cleaned_text = clean_srt_content(sub.content).replace('\n', ' ').strip()
            if not cleaned_text:
                continue
            processed.append(
                {
                    "id": sub.index if sub.index is not None else order + 1,
                    "order": order,
                    "start": sub.start.total_seconds(),
                    "end": sub.end.total_seconds(),
                    "text": cleaned_text,
                }
            )
        full_text = " ".join(entry["text"] for entry in processed)
        print(f"  -> SRT fájl sikeresen betöltve: {os.path.basename(filepath)} ({len(processed)} felirat).")
        return processed, full_text
    except Exception as e:
        print(f"  -> Figyelmeztetés: Az SRT fájl olvasása sikertelen ({e}).")
        return [], ""


def create_smart_chunks(
    segments: List[Dict[str, Any]],
    min_size: int = 50,
    max_size: int = 100,
    gap_threshold: float = 5.0,
) -> List[List[Dict[str, Any]]]:
    chunks: List[List[Dict[str, Any]]] = []
    current_pos = 0
    while current_pos < len(segments):
        chunk_start = current_pos
        if (len(segments) - chunk_start) <= min_size:
            best_split_point = len(segments)
        else:
            best_split_point = -1
            search_end = min(chunk_start + max_size, len(segments) - 1)
            for i in range(chunk_start + min_size - 1, search_end):
                try:
                    if segments[i + 1].get('start', 0) - segments[i].get('end', 0) >= gap_threshold:
                        best_split_point = i + 1
                        break
                except (TypeError, KeyError):
                    continue
            if best_split_point == -1:
                max_gap = -1.0
                best_split_point = min(chunk_start + max_size, len(segments))
                for i in range(chunk_start + min_size - 1, search_end):
                    try:
                        gap = segments[i + 1].get('start', 0) - segments[i].get('end', 0)
                        if gap > max_gap:
                            max_gap = gap
                            best_split_point = i + 1
                    except (TypeError, KeyError):
                        continue
        chunks.append(segments[chunk_start:best_split_point])
        current_pos = best_split_point
    return chunks


# --- IDŐALAPÚ ÉS LLM-ES ILLesztési SEGÉDFÜGGVÉNYEK --------------------------------

def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    td = timedelta(seconds=seconds)
    total_seconds = td.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def collect_srt_segments_for_batch(
    batch_segments: List[Dict[str, Any]],
    srt_entries: List[Dict[str, Any]],
    base_margin: float = 6.0,
    max_margin: float = 40.0,
) -> List[Dict[str, Any]]:
    if not batch_segments or not srt_entries:
        return []

    valid_starts = [seg.get('start') for seg in batch_segments if isinstance(seg.get('start'), (int, float))]
    valid_ends = [seg.get('end') for seg in batch_segments if isinstance(seg.get('end'), (int, float))]
    if not valid_starts or not valid_ends:
        return []

    batch_start = min(valid_starts)
    batch_end = max(valid_ends)

    margin = base_margin
    while margin <= max_margin:
        context_start = max(0.0, batch_start - margin)
        context_end = batch_end + margin
        candidates = [
            entry for entry in srt_entries
            if entry['end'] >= context_start and entry['start'] <= context_end
        ]
        if candidates:
            return candidates
        margin *= 1.8

    # Ha továbbra sincs találat, próbáljuk az egész SRT-t.
    return srt_entries


def build_prompt_rows_for_segments(batch_segments: List[Dict[str, Any]]) -> str:
    rows = []
    for idx, segment in enumerate(batch_segments, start=1):
        start_ts = format_timestamp(float(segment.get('start', 0.0)))
        end_ts = format_timestamp(float(segment.get('end', segment.get('start', 0.0))))
        text = segment.get('text', '').strip().replace('\n', ' ')
        rows.append(f"{idx}. [{start_ts} – {end_ts}] {text}")
    return "\n".join(rows)


def build_prompt_rows_for_srt(entries: List[Dict[str, Any]]) -> str:
    rows = []
    for entry in entries:
        start_ts = format_timestamp(entry['start'])
        end_ts = format_timestamp(entry['end'])
        rows.append(f"{entry['id']}# [{start_ts} – {end_ts}] {entry['text']}")
    return "\n".join(rows)


SENTENCE_END_PATTERN = re.compile(r'[\.!?…]+[)"\'»\]]*')
BOUNDARY_FALLOFF = 60


def normalize_entry_text(raw_text: str) -> str:
    return re.sub(r'\s+', ' ', raw_text.strip())


def build_concatenated_text_for_split(
    entries: List[Dict[str, Any]],
) -> Tuple[str, List[int], List[int]]:
    parts: List[str] = []
    sentence_boundaries: List[int] = []
    entry_boundaries: List[int] = []
    cursor = 0

    for entry in entries:
        normalized = normalize_entry_text(entry.get('text', ''))
        if not normalized:
            continue
        if parts:
            parts.append(' ')
            cursor += 1
        parts.append(normalized)
        cursor += len(normalized)
        entry_boundaries.append(cursor)

        offset_base = cursor - len(normalized)
        for match in SENTENCE_END_PATTERN.finditer(normalized):
            boundary = offset_base + match.end()
            if 0 < boundary < cursor + 1:
                sentence_boundaries.append(boundary)

    combined_text = ''.join(parts)
    text_len = len(combined_text)
    if text_len == 0:
        return combined_text, [], []

    sentence_boundaries = sorted(
        set(boundary for boundary in sentence_boundaries if 0 < boundary < text_len)
    )
    entry_boundaries = sorted(
        set(boundary for boundary in entry_boundaries if 0 < boundary < text_len)
    )
    return combined_text, sentence_boundaries, entry_boundaries


def _find_whitespace_boundary(text: str, approx_index: int) -> int:
    if approx_index <= 0:
        return 0
    if approx_index >= len(text):
        return len(text)

    for idx in range(approx_index, len(text)):
        if text[idx].isspace():
            return idx
    for idx in range(approx_index, 0, -1):
        if text[idx - 1].isspace():
            return idx
    return len(text)


def _pick_nearest_boundary(boundaries: List[int], approx: int, window: int) -> Optional[int]:
    if not boundaries:
        return None
    idx = bisect_left(boundaries, approx)
    candidates: List[int] = []
    if idx < len(boundaries):
        candidates.append(boundaries[idx])
    if idx > 0:
        candidates.append(boundaries[idx - 1])

    best_candidate: Optional[int] = None
    best_distance: Optional[int] = None
    for candidate in candidates:
        distance = abs(candidate - approx)
        if distance <= window:
            if (
                best_candidate is None
                or distance < best_distance
                or (distance == best_distance and candidate > best_candidate)
            ):
                best_candidate = candidate
                best_distance = distance
    return best_candidate


def _choose_split_boundary(
    text: str,
    approx: int,
    sentence_boundaries: List[int],
    entry_boundaries: List[int],
    previous_boundary: int,
    segments_left: int,
) -> int:
    text_len = len(text)
    min_allowed = previous_boundary + 1
    max_allowed = text_len - max(1, segments_left)
    if max_allowed < min_allowed:
        max_allowed = min_allowed

    approx = max(min_allowed, min(approx, max_allowed))
    sentence_window = max(12, min(BOUNDARY_FALLOFF, text_len // 20 or 12))
    entry_window = max(18, min(BOUNDARY_FALLOFF, text_len // 15 or 18))

    boundary = _pick_nearest_boundary(sentence_boundaries, approx, sentence_window)
    if boundary is None:
        boundary = _pick_nearest_boundary(entry_boundaries, approx, entry_window)
    if boundary is None:
        boundary = _find_whitespace_boundary(text, approx)

    boundary = max(min_allowed, min(boundary, max_allowed))
    return boundary


def split_text_by_weights(
    text: str,
    weights: List[float],
    sentence_boundaries: List[int],
    entry_boundaries: List[int],
) -> List[str]:
    if not text:
        return [''] * len(weights)
    if not weights:
        return [text.strip()]

    total_weight = sum(weights)
    if total_weight <= 0:
        total_weight = float(len(weights))
        weights = [1.0] * len(weights)

    cumulative_ratios: List[float] = []
    running = 0.0
    for weight in weights[:-1]:
        running += weight
        cumulative_ratios.append(running / total_weight)

    text_len = len(text)
    boundaries: List[int] = []
    previous_boundary = 0
    for idx, ratio in enumerate(cumulative_ratios, start=1):
        approx_index = int(round(ratio * text_len))
        boundary = _choose_split_boundary(
            text,
            approx_index,
            sentence_boundaries,
            entry_boundaries,
            previous_boundary,
            len(weights) - idx,
        )
        boundaries.append(boundary)
        previous_boundary = boundary

    parts: List[str] = []
    previous = 0
    for boundary in boundaries:
        parts.append(text[previous:boundary].strip())
        previous = boundary
    parts.append(text[previous:].strip())

    while len(parts) < len(weights):
        parts.append('')
    if len(parts) > len(weights):
        overflow = parts[len(weights):]
        parts = parts[:len(weights)]
        remainder = ' '.join(chunk for chunk in overflow if chunk).strip()
        if remainder:
            parts[-1] = (parts[-1] + ' ' + remainder).strip()
    return parts


def split_entry_text_portion(text: str, take_ratio: float) -> Tuple[str, str]:
    clean_text = text.strip()
    if not clean_text:
        return '', ''
    if take_ratio <= 0.02:
        return '', clean_text
    if take_ratio >= 0.98 or (len(clean_text) <= 24 and take_ratio >= 0.9):
        return clean_text, ''

    text_len = len(clean_text)
    approx_index = max(1, min(text_len - 1, int(round(text_len * take_ratio))))
    sentence_boundaries = sorted(
        match.end()
        for match in SENTENCE_END_PATTERN.finditer(clean_text)
        if 0 < match.end() < text_len
    )
    comma_boundaries = sorted(
        {
            match.start() + 1
            for match in re.finditer(r',[\\s]+', clean_text)
            if 0 < (match.start() + 1) < text_len
        }
    )
    sentence_window = max(6, min(BOUNDARY_FALLOFF, text_len // 3 or 6))
    boundary = _pick_nearest_boundary(sentence_boundaries, approx_index, sentence_window)
    if boundary is None:
        boundary = _pick_nearest_boundary(comma_boundaries, approx_index, sentence_window)
    if boundary is None:
        boundary = _find_whitespace_boundary(clean_text, approx_index)

    boundary = max(1, min(boundary, text_len - 1))
    allocated = clean_text[:boundary].strip()
    remaining = clean_text[boundary:].strip()
    if not allocated and text_len > 1:
        boundary = max(1, min(text_len - 1, approx_index))
        allocated = clean_text[:boundary].strip()
        remaining = clean_text[boundary:].strip()
    if remaining:
        words = allocated.split()
        if words:
            last_raw = words[-1]
            last_alpha = last_raw.strip(".,;:!?…\"'”»“«")
            if len(last_alpha) == 1:
                words = words[:-1]
                allocated = ' '.join(words).strip()
                remaining = f"{last_alpha} {remaining}".strip()
    if remaining and len(remaining) <= 3:
        allocated = f"{allocated} {remaining}".strip()
        remaining = ''
    return allocated, remaining


def align_segments_via_time(
    batch_segments: List[Dict[str, Any]],
    srt_context: List[Dict[str, Any]],
) -> List[str]:
    if not batch_segments:
        return []

    entry_states: List[Dict[str, Any]] = []
    for entry in srt_context:
        normalized_text = normalize_entry_text(entry.get('text', ''))
        if not normalized_text:
            continue
        entry_states.append(
            {
                "start": float(entry['start']),
                "end": float(entry['end']),
                "cursor": float(entry['start']),
                "text": normalized_text,
            }
        )

    results: List[str] = []
    entry_idx = 0

    for segment in batch_segments:
        seg_start = float(segment.get('start', 0.0))
        seg_end = float(segment.get('end', seg_start))
        if seg_end <= seg_start:
            seg_end = seg_start + 0.001
        segment_duration = max(0.001, seg_end - seg_start)

        collected_parts: List[str] = []
        while entry_idx < len(entry_states):
            state = entry_states[entry_idx]

            if state['end'] <= seg_start + 1e-6:
                entry_idx += 1
                continue

            if state['start'] >= seg_end - 1e-6 and not collected_parts:
                break

            if state['cursor'] >= state['end'] - 1e-6 or not state['text']:
                entry_idx += 1
                continue

            overlap_start = max(seg_start, state['cursor'])
            overlap_end = min(seg_end, state['end'])
            if overlap_end <= overlap_start:
                if state['start'] >= seg_end - 1e-6:
                    break
                entry_idx += 1
                continue

            remaining_duration = max(0.0, state['end'] - state['cursor'])
            take_duration = max(0.0, min(overlap_end - overlap_start, remaining_duration))
            if take_duration <= 1e-6 or remaining_duration <= 1e-6:
                entry_idx += 1
                continue

            take_ratio = min(1.0, max(0.0, take_duration / remaining_duration))
            segment_overlap_ratio = min(1.0, max(0.0, take_duration / segment_duration))

            if (
                take_ratio < 0.12
                and segment_overlap_ratio < 0.2
                and overlap_start <= state['cursor'] + 1e-6
                and overlap_end < state['end'] - 1e-3
            ):
                break

            if take_ratio >= 0.85 or (take_ratio >= 0.5 and segment_overlap_ratio >= 0.55):
                allocated_text = state['text']
                leftover_text = ''
            else:
                allocated_text, leftover_text = split_entry_text_portion(state['text'], take_ratio)

            if allocated_text:
                collected_parts.append(allocated_text)

            state['text'] = leftover_text
            state['cursor'] = overlap_start + take_duration

            if not state['text'] or state['cursor'] >= state['end'] - 1e-6:
                entry_idx += 1

            if overlap_end >= seg_end - 1e-6:
                break

        results.append(' '.join(part for part in collected_parts if part).strip())

    if len(results) < len(batch_segments):
        results.extend([''] * (len(batch_segments) - len(results)))
    elif len(results) > len(batch_segments):
        results = results[:len(batch_segments)]

    return results


def proportional_split_by_time(
    batch_segments: List[Dict[str, Any]],
    srt_context: List[Dict[str, Any]],
) -> List[str]:
    if not batch_segments:
        return []
    combined_text, sentence_boundaries, entry_boundaries = build_concatenated_text_for_split(srt_context)
    if not combined_text:
        return [''] * len(batch_segments)

    durations: List[float] = []
    for segment in batch_segments:
        start = float(segment.get('start', 0.0))
        end = float(segment.get('end', start))
        duration = max(0.01, end - start)
        durations.append(duration)
    return split_text_by_weights(combined_text, durations, sentence_boundaries, entry_boundaries)


SYSTEM_PROMPT = (
    "You are an expert bilingual subtitle editor. Your job is to align a high-quality translation "
    "with the structure of an English ASR transcript.\n"
    "Always keep the meaning of the Hungarian text, only re-distribute it to match the English segment boundaries."
)


def extract_json_from_text(raw_text: str) -> Optional[Any]:
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    array_match = re.search(r'(\[\s*{.*}\s*\])', raw_text, re.DOTALL)
    if array_match:
        candidate = array_match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None


def parse_alignment_response(raw_text: str, expected_count: int) -> Optional[List[str]]:
    payload = extract_json_from_text(raw_text)
    if not isinstance(payload, list):
        return None
    results = [''] * expected_count
    for item in payload:
        if not isinstance(item, dict):
            return None
        idx = item.get('index') or item.get('segment_index') or item.get('target_index')
        text = item.get('text') or item.get('value')
        if idx is None or text is None:
            return None
        try:
            idx_int = int(idx)
        except (ValueError, TypeError):
            return None
        if not (1 <= idx_int <= expected_count):
            return None
        cleaned_text = str(text).strip()
        if not cleaned_text:
            return None
        if results[idx_int - 1]:
            return None
        results[idx_int - 1] = cleaned_text
    if any(not entry for entry in results):
        return None
    return results


def attempt_llm_alignment(
    client: OpenAI,
    model: str,
    batch_segments: List[Dict[str, Any]],
    srt_context: List[Dict[str, Any]],
    suggested_lines: List[str],
) -> Optional[List[str]]:
    english_overview = build_prompt_rows_for_segments(batch_segments)
    srt_overview = build_prompt_rows_for_srt(srt_context)
    suggestions = "\n".join(f"{idx + 1}. {line}" for idx, line in enumerate(suggested_lines)) if suggested_lines else "n/a"

    user_prompt = (
        "### English ASR segments (target structure)\n"
        f"{english_overview}\n\n"
        "### Hungarian subtitle segments (source material)\n"
        f"{srt_overview}\n\n"
        "### Suggested alignment from time-based heuristic (adjust if needed)\n"
        f"{suggestions}\n\n"
        "TASK:\n"
        "Return a JSON array where each item has the keys 'index' (1-based, matching the English list) and 'text' "
        "(the Hungarian content for that index). Optionally you may add 'source_ids' with the SRT ids you used.\n"
        "Keep the Hungarian phrasing; only split or merge sentences so they align with the English segment boundaries.\n"
        "Every English segment must appear exactly once in the output array, in any order."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
    except OpenAIError as e:
        print(f"  HIBA: API hiba az igazítás kérés közben: {e}")
        return None

    raw_content = response.choices[0].message.content if response.choices else ""
    parsed = parse_alignment_response(raw_content, len(batch_segments))
    if parsed is None:
        print("  Figyelmeztetés: Az LLM kimenete nem volt értelmezhető JSON formátumban.")
    return parsed


def align_and_segment_batch(
    client: OpenAI,
    batch_segments: List[Dict[str, Any]],
    target_lang_srt_entries: List[Dict[str, Any]],
    model: str,
    stream: bool,
    batch_id_str: str,
) -> List[str]:
    if not batch_segments:
        return []

    srt_context = collect_srt_segments_for_batch(batch_segments, target_lang_srt_entries)
    if not srt_context:
        print(f"  HIBA: Nem található megfelelő SRT kontextus a(z) [{batch_id_str}] csoporthoz.")
        return [''] * len(batch_segments)

    time_aligned_lines = align_segments_via_time(batch_segments, srt_context)
    if len(time_aligned_lines) != len(batch_segments):
        padding = len(batch_segments) - len(time_aligned_lines)
        time_aligned_lines.extend([''] * padding)

    non_empty = sum(1 for line in time_aligned_lines if line)
    coverage_ratio = non_empty / max(1, len(time_aligned_lines))
    print(f"  -> Időalapú igazítás lefedettsége: {coverage_ratio * 100:.1f}%")

    proportional_lines: List[str] = []
    if coverage_ratio < 0.6:
        proportional_lines = proportional_split_by_time(batch_segments, srt_context)

    llm_result: Optional[List[str]] = None
    if coverage_ratio < 0.92 and model:
        suggestion_lines = time_aligned_lines if any(time_aligned_lines) else proportional_lines
        llm_result = attempt_llm_alignment(client, model, batch_segments, srt_context, suggestion_lines)

    final_lines: List[str]
    if llm_result and len(llm_result) == len(batch_segments):
        final_lines = llm_result
    elif coverage_ratio >= 0.4:
        final_lines = time_aligned_lines
    else:
        if proportional_lines:
            print(f"  Figyelmeztetés: A(z) [{batch_id_str}] csoportnál időalapú igazítás hiányos, arányos felosztási tartalék használata.")
            final_lines = proportional_lines
        else:
            final_lines = time_aligned_lines

    if proportional_lines and final_lines is not proportional_lines:
        final_lines = [
            line if line.strip() else proportional_lines[idx]
            for idx, line in enumerate(final_lines)
        ]

    if stream:
        for line in final_lines:
            print(f"    + {line}")
    return final_lines


# --- FŐ FOLYAMAT -----------------------------------------------------------------

def main(
    project_name: str,
    input_lang: str,
    output_lang: str,
    auth_key_arg: Optional[str],
    context: Optional[str],
    model: str,
    stream: bool,
    allow_sensitive_content: bool,
) -> None:
    auth_key = auth_key_arg
    if auth_key:
        save_api_key(auth_key)
    else:
        print("API kulcs parancssorból nincs megadva, betöltés a keyholder.json fájlból...")
        auth_key = load_api_key()

    if not auth_key:
        print("\nHIBA: Nincs elérhető OpenAI API kulcs.")
        sys.exit(1)

    print("OpenAI API kulcs sikeresen beállítva.")

    config = load_config()
    if not config:
        sys.exit(1)

    try:
        workdir = config['DIRECTORIES']['workdir']
        subdirs = config['PROJECT_SUBDIRS']
        input_dir = os.path.join(workdir, project_name, subdirs['separated_audio_speech'])
        output_dir = os.path.join(workdir, project_name, subdirs['translated'])
        upload_dir = os.path.join(workdir, project_name, subdirs['upload'])
    except KeyError as e:
        print(f"Hiba: Hiányzó kulcs a config.json fájlban: {e}")
        return

    client = OpenAI(api_key=auth_key)
    input_filename, error = find_json_file(input_dir)
    if error:
        return

    input_filepath = os.path.join(input_dir, input_filename)
    print(f"Bemeneti fájl feldolgozása: {input_filepath}")
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, input_filename)

    print(f"\nCélnyelvi (magyar) SRT keresése a(z) '{upload_dir}' mappában ('{output_lang.upper()}')...")
    target_srt_filepath = find_srt_context_file(upload_dir, output_lang)
    target_srt_entries, _ = load_srt_file(target_srt_filepath)

    if not target_srt_entries:
        print("HIBA: A célnyelvi (magyar) SRT fájl elengedhetetlen ehhez a művelethez, de nem található. A szkript leáll.")
        return

    progress_filename = f"{os.path.splitext(input_filename)[0]}.progress.json"
    progress_filepath = os.path.join(output_dir, progress_filename)
    progress: Dict[str, str] = {}
    if os.path.exists(progress_filepath):
        try:
            with open(progress_filepath, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            print(f"\nHaladási fájl betöltve: {progress_filepath} ({len(progress)} szegmens már feldolgozva).")
        except (json.JSONDecodeError, IOError):
            progress = {}

    with open(input_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'segments' not in data or not isinstance(data['segments'], list):
        print("Hiba: JSON 'segments' kulcs hiányzik.")
        return

    unprocessed_segments: List[Dict[str, Any]] = []
    for i, segment in enumerate(data['segments']):
        if segment.get('text', '').strip() and str(i) not in progress:
            segment['original_index'] = i
            unprocessed_segments.append(segment)

    if not unprocessed_segments:
        print("Nincs új, feldolgozandó szegmens.")
    else:
        print(f"\n{len(unprocessed_segments)} új, feldolgozandó szegmens található.")
        batches = create_smart_chunks(unprocessed_segments)
        total_batches = len(batches)
        print(f"Intelligens csoportképzés befejezve. {total_batches} új csoportot kell feldolgozni.")

        for i, batch in enumerate(batches):
            print(f"\n[{i + 1}/{total_batches}] fő csoport feldolgozása...")
            aligned_batch_lines = align_and_segment_batch(
                client,
                batch,
                target_srt_entries,
                model,
                stream,
                f"{i + 1}",
            )

            if aligned_batch_lines is None:
                print("\nA feldolgozási folyamat megszakadt.")
                return

            for segment_obj, aligned_line in zip(batch, aligned_batch_lines):
                progress[str(segment_obj['original_index'])] = aligned_line

            with open(progress_filepath, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            print(f"  Haladás elmentve a(z) {progress_filepath} fájlba.")
            time.sleep(1)

    print("\nAz összes szükséges szegmens igazítása elkészült.")
    for index_str, line in progress.items():
        data['segments'][int(index_str)]['translated_text'] = line
    for segment in data['segments']:
        segment.setdefault('translated_text', '')
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nFeldolgozás befejezve. A kiegészített fájl a(z) '{output_filepath}' helyre mentve.")

    if os.path.exists(progress_filepath):
        os.remove(progress_filepath)
        print(f"Az ideiglenes haladási fájl ({progress_filepath}) törölve.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Meglévő fordítás (SRT) újra-szegmentálása egy ASR JSON időzítése alapján – továbbfejlesztett igazítással.',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument('-project_name', required=True, help='A workdir-en belüli projektmappa neve.')
    parser.add_argument('-input_language', default='EN', help='A bemeneti nyelv kódja (pl. EN, HU). Alapértelmezett: EN')
    parser.add_argument('-output_language', default='HU', help='A kimeneti nyelv kódja (pl. EN, HU), aminek az SRT-jét keresi. Alapértelmezett: HU')
    parser.add_argument('-auth_key', required=False, help='Az OpenAI API hitelesítési kulcs. Ha megadjuk, elmentődik a keyholder.json-ba.\nHa nem adjuk meg, onnan próbálja betölteni.')
    parser.add_argument('-context', required=False, help='Ez az argumentum ennél a módszernél figyelmen kívül van hagyva.')
    parser.add_argument('-model', default='gpt-4o', required=False, help='A használni kívánt OpenAI modell neve. Alapértelmezett: gpt-4o')
    parser.add_argument('-stream', action='store_true', help='Bekapcsolja a valós idejű kimenetet.')
    parser.add_argument('--allow-sensitive-content', action=argparse.BooleanOptionalAction, default=True, help='Ez az argumentum ennél a módszernél figyelmen kívül van hagyva.')
    add_debug_argument(parser)

    args = parser.parse_args()
    configure_debug_mode(args.debug)

    main(
        args.project_name,
        args.input_language,
        args.output_language,
        args.auth_key,
        args.context,
        args.model,
        args.stream,
        args.allow_sensitive_content,
    )
