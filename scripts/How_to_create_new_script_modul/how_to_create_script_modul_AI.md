# AI Brief – Build a New Script Module

Follow these instructions exactly to deliver a production-ready module. The module **must** consist of three files sharing the same base name:
1. `<name>.py`
2. `<name>.json`
3. `<name>_help.md`

---

## Core Principles

- Resolve project-specific paths through the repository `config.json`; never hard-code absolute locations.
- Require a project selector argument (usually `-p/--project-name`) unless the workflow truly operates globally.
- Keep the CLI definition, JSON descriptor, and `_help.md` perfectly in sync.
- Always wire in `add_debug_argument` / `configure_debug_mode` so verbosity can be toggled consistently.
- Implement idempotent processing: rerunning should not duplicate outputs or corrupt previous results.
- Log the key configuration (project, input/output directories, notable flags) during start-up to aid diagnostics.

---

## 1. Python Script (`.py`)

1. **Bootstrap**
   - Import `argparse`, `json`, `logging`, `sys`, `Path` and reuse utilities:  
     `from tools.debug_utils import add_debug_argument, configure_debug_mode`.
   - Implement `get_project_root()` that walks upwards until `config.json` is found; abort with `FileNotFoundError` if missing.
   - Implement `load_config()` returning `(config_dict, project_root_path)`; on error prints and `sys.exit(1)`.

2. **CLI**
   - Build an `ArgumentParser` with a required project argument (`-p/--project-name` or domain specific equivalent).
   - Add every extra option/flag required by the feature; assign `type`, `default`, and `help`.
   - Append `add_debug_argument(parser)` and call `configure_debug_mode(args.debug)` right after parsing.

3. **Paths & Validation**
   - Resolve project-specific paths using `config["DIRECTORIES"]` and `config["PROJECT_SUBDIRS"]`.
   - Validate existence of every input directory/file; on failure print a clear error and `sys.exit(1)`.

4. **Core Logic**
   - Implement the processing pipeline; keep it idempotent and deterministic.
   - Log key steps (inputs discovered, outputs written, skipped items, errors).
   - Wrap risky blocks with `try/except`; on fatal errors log and exit cleanly.
   - Use type hints; when emitting localized JSON, call `json.dump(..., ensure_ascii=False)`.
   - If a required dependency is missing, surface a clear error with installation guidance.

5. **Main Guard**
   - Provide `main()` and protect it with `if __name__ == "__main__": main()`.

---

## 2. Configuration Descriptor (`.json`)

Create `scripts/<subdir>/<name>.json` with the structure:
```json
{
  "enviroment": "<conda_env>",
  "script": "<relative/path/to/name.py>",
  "description": "<one sentence summary>",
  "required": [ ... ],
  "optional": [ ... ],
  "api": "<service>" // optional
}
```

- Each parameter object mirrors the `argparse` definition:
  ```json
  {
    "name": "<arg_dest>",
    "flags": ["-x", "--example"],   // omit for positional args
    "type": "option|flag|positional|config_option",
    "default": <json_value_or_null>
  }
  ```
- Include both positive and negative flags if the CLI supports them (e.g. `--feature` and `--no-feature`), with accurate defaults.
- Defaults must be valid JSON scalars (`null`, number, string, boolean).
- Keep the `script` field aligned with the actual file path.
- Update the central `scripts/scripts.json` registry when a new module or parameter set must be exposed to the launcher.
- Ensure every option listed here exists in the Python script and vice versa.

---

## 3. Help Document (`_help.md`)

Create a Markdown file alongside the script:

```
# <name> – short tagline
**Runtime:** `<env>`  
**Entry point:** `<relative path>`

## Overview
<goal, prerequisites, workflow>

## Required Parameters
- `<flag>`: <purpose, default>

## Optional Parameters
- `<flag>`: <purpose, default>

## Outputs
- Describe generated files and locations.

## Error Handling / Tips
- List failure modes, required dependencies, keyholder usage, etc.
```

Keep the content concise, actionable, and consistent with the script and JSON.
Document default values and note how paired flags interact (e.g., `--no-backup` turns off backups).

---

## 4. Consistency Checklist

Before finishing:

- [ ] `.py`, `.json`, and `_help.md` all exist under the correct `scripts/<category>/` subdirectory.
- [ ] The CLI, JSON, and help file list identical arguments, defaults, and behavior.
- [ ] All path resolutions rely on `config.json`; error messages are user friendly.
- [ ] Logging works and `--debug` toggles verbosity.
- [ ] External dependencies are checked with clear guidance when missing.
- [ ] Running `python <path>/<name>.py --help` produces output that matches both JSON and documentation.
- [ ] API credentials (if needed) are stored and retrieved via `keyholder.json`.
