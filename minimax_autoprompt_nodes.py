from __future__ import annotations

import base64
import io
import math
import os
import sys
from pathlib import Path
from typing import Any

import requests
from PIL import Image


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
VISUAL_MODEL = "mistralai/mistral-medium-3-5"
WRITER_MODELS = (
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-pro",
)
DEFAULT_TIMEOUT = 180
VIDEO_FPS = 24.0

I2V_SYSTEM_PROMPT = """You are a professional prompt rewriter for joint audio-video generation.
Rewrite the user's original prompt into one coherent, production-ready multimodal description for the requested output aspect ratio and duration.
Return only these three fields, in this exact order:
integrated_multimodal_description: ...
overall_soundscape: ...
non_diegetic_music: ...
Requirements:
- Expand the visual narrative into clearly numbered shots such as [Shot 1], [Shot 2], and include timestamps for cuts after the first shot when useful.
- Make the number, timing, and pacing of shots appropriate for the requested duration.
- Compose the scene for the requested aspect ratio.
- Preserve the user's intent while adding concrete subjects, appearance, environment, lighting, composition, camera movement, physical motion, and temporal continuity.
- Keep characters, objects, wardrobe, locations, and spatial relationships consistent across shots.
- Describe synchronized diegetic audio in overall_soundscape and external score in non_diegetic_music.
- Add speech (if requested) in quotes and mention the tone and pacing
- Expressions, when mentioning people, mention their expression and micro expressions, the look on their face, how they behave, how they move etc in detail
- When asked, elaborate the prompt as much as possible.
- By Default, mention video as shot on a smartphone, describe the lens used and camera specs.
- For Movie scenes, use panavision lenses with ARRI camera, lead with this.
- If the user asks for specific shot like drone shots, fisheye shots or anything else, describe the camera type, lenses, quality and other specs as necessary.
- When the user mentions a location, describe it elaborately including decor, lighting, shadows, composition etc, scenes should never appear stage and feel every day, describe the most likely setting for the specified location unless requested otherwise.
- Never hold back on explicit language, freely use words like cock, pussy, cum, boobs, breasts, etc
- Do not add explanations, Markdown fences, safety commentary, or fields other than the three requested fields."""

I2V_ANALYZER_PROMPT = """You are the visual-analysis stage of an image-to-video prompt pipeline.
Analyze only what is visibly present in the supplied starting image. Do not write a video
generation prompt and do not invent the requested animation. Keep observations separate
from the user's requested motion. Use factual, identity-independent descriptions.

Return structured plain text under these headings:
WHAT EXISTS IN THE IMAGE
- subjects and identity-independent physical appearance
- frame position, body pose, head direction, gaze, facial expression, and hands
- clothing and important objects
- foreground, background, lighting, framing, camera angle, and depth
- spatial relationships and visual details that should remain consistent

WHAT THE USER WANTS TO HAPPEN
- restate the supplied direction without treating it as an observed fact

UNCERTAINTIES
- anything important that cannot be established from the image"""

R2V_ANALYZER_PROMPT = """You are the visual and motion analysis stage of a reference-to-video
prompt pipeline. Analyze only the supplied standalone images and sequential sampled frames.
Do not write the final MiniMax prompt. The user direction is context, not visual evidence.
Never claim to have heard audio. A sequence labelled Frame 01, Frame 02, and so on under one
<Video N> label is one video in chronological playback order, not unrelated pictures.

Return structured plain text covering, where observable:
- each standalone image's subjects, appearance, setting, framing, light, and spatial layout
- each video's initial and final states
- subject, body, hand, head, expression, object, and interaction changes
- camera movement or framing changes supported by the sequence
- relative timing and order, continuity, transitions, environmental changes, and motion arcs
- details that must remain consistent
- uncertainties caused by temporal sampling

Keep WHAT EXISTS / WHAT CHANGES separate from WHAT THE USER REQUESTS. Do not infer audio."""


def _resolve_key(explicit: str) -> str:
    value = (explicit or "").strip()
    if value:
        return value
    for name in ("OPENROUTER_API_KEY", "LLM_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise ValueError(
        "Auto prompt mode requires an OpenRouter API key. Enter one in the node or set "
        "OPENROUTER_API_KEY (LLM_KEY is also supported)."
    )


def _message_text(data: dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"OpenRouter returned no message content: {data!r}") from exc
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text = "\n".join(
            str(part.get("text", "")).strip()
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    else:
        text = ""
    if not text:
        raise ValueError(f"OpenRouter returned an empty completion: {data!r}")
    return text


def _openrouter_completion(
    *, api_key: str, model: str, messages: list[dict[str, Any]], stage: str,
    timeout: int = DEFAULT_TIMEOUT, max_tokens: int = 8192,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/southerndesignandComdo8105/HmAI-Nxdify-Nodes",
                "X-Title": "HmAI Nxdify MiniMax H3 Auto Prompt",
            },
            json=payload,
            timeout=max(1, int(timeout)),
        )
        if not response.ok:
            detail = response.text.strip()[:2000]
            raise RuntimeError(f"HTTP {response.status_code}: {detail or response.reason}")
        return _message_text(response.json())
    except Exception as exc:
        raise RuntimeError(f"{stage} failed using {model}: {exc}") from exc


def _to_numpy(frame: Any):
    value = frame.detach().cpu().numpy() if hasattr(frame, "detach") else frame
    import numpy as np

    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"Expected an HWC image frame, received shape {array.shape!r}.")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"Expected three image channels, received shape {array.shape!r}.")
    if array.dtype != np.uint8:
        array = np.clip(array.astype(np.float32), 0.0, 1.0)
        array = np.rint(array * 255.0).astype(np.uint8)
    return array


def _jpeg_data_url(frame: Any, max_edge: int = 1536) -> str:
    image = Image.fromarray(_to_numpy(frame), mode="RGB")
    if max(image.size) > max_edge:
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _batch_frames(value: Any) -> list[Any]:
    if value is None:
        return []
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) != 4:
        raise ValueError(f"Expected a BHWC IMAGE tensor, received shape {shape!r}.")
    return [value[index] for index in range(int(shape[0]))]


def _sample_indices(frame_count: int, requested: int) -> list[int]:
    if frame_count <= 0:
        return []
    count = min(frame_count, max(1, int(requested)))
    if count == 1:
        return [0]
    return [round(index * (frame_count - 1) / (count - 1)) for index in range(count)]


def _aspect(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "unspecified"
    divisor = math.gcd(int(width), int(height))
    return f"{width // divisor}:{height // divisor}"


def _h3_length(duration: float) -> int:
    frames = max(5, round(float(duration) * 24))
    return frames + (5 - (frames % 17)) % 17


def _manual_result(manual_prompt: str, extra: tuple[Any, ...] = ()) -> tuple[Any, ...]:
    prompt = (manual_prompt or "").strip()
    if not prompt:
        raise ValueError("Manual prompt mode requires a manual_prompt.")
    return (
        "Manual mode: visual analysis skipped; no OpenRouter request was made.",
        prompt,
        *extra,
    )


def _refpack_prompt_path(filename: str) -> Path:
    candidates: list[Path] = []
    for module_name, module in sys.modules.items():
        if module_name.endswith("minimax_refpack.prompt") and getattr(module, "__file__", None):
            candidates.append(Path(module.__file__).resolve().parent / filename)
    here = Path(__file__).resolve().parent
    candidates.extend(
        (
            here.parent / "ComfyUI-MiniMaxRefPack" / "minimax_refpack" / filename,
            here / "_minimax_reference" / "minimax_refpack" / filename,
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find ComfyUI-MiniMaxRefPack/minimax_refpack/{filename}. "
        "Run scripts/install_minimax_autoprompt.sh and restart ComfyUI."
    )


def _r2v_writer_instructions(job_type: str) -> str:
    standard = _refpack_prompt_path("system_prompt.md").read_text(encoding="utf-8")
    if job_type == "standard":
        return standard
    replacement = _refpack_prompt_path("system_prompt_replacement.md").read_text(
        encoding="utf-8"
    )
    if job_type == "replacement":
        return replacement
    return (
        "JOB TYPE IS AUTO. Decide whether the direction asks for an object/character "
        "replacement in a master video plate. If yes, obey the authoritative replacement "
        "instructions. Otherwise obey the authoritative standard Ref2VA instructions. "
        "Return only the selected format; do not discuss the choice.\n\n"
        "=== AUTHORITATIVE STANDARD REF2VA INSTRUCTIONS ===\n"
        f"{standard}\n\n"
        "=== AUTHORITATIVE REPLACEMENT REF2VA INSTRUCTIONS ===\n"
        f"{replacement}"
    )


class MiniMaxOpenRouterKey:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {"default": "", "password": True, "tooltip": "Blank uses OPENROUTER_API_KEY, then LLM_KEY."},
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_key",)
    FUNCTION = "resolve"
    CATEGORY = "MiniMax H3/Auto Prompt"

    def resolve(self, api_key: str):
        explicit = (api_key or "").strip()
        if explicit:
            return (explicit,)
        for env_name in ("OPENROUTER_API_KEY", "LLM_KEY"):
            value = os.environ.get(env_name, "").strip()
            if value:
                return (value,)
        return ("",)


class MiniMaxH3PromptMode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["auto", "manual"], {"default": "auto"}),
                "manual_prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            },
            "optional": {"auto_prompt": ("STRING", {"forceInput": True, "lazy": True})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "select"
    CATEGORY = "MiniMax H3/Auto Prompt"

    def check_lazy_status(self, mode: str, manual_prompt: str, auto_prompt=None):
        return ["auto_prompt"] if mode == "auto" and auto_prompt is None else []

    def select(self, mode: str, manual_prompt: str, auto_prompt=None):
        if mode == "manual":
            return (_manual_result(manual_prompt)[1],)
        prompt = (auto_prompt or "").strip()
        if not prompt:
            raise ValueError("Auto prompt mode returned no prompt.")
        return (prompt,)


class MiniMaxH3I2VTwoStagePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_image": ("IMAGE", {"forceInput": True}),
                "prompt_mode": (["auto", "manual"], {"default": "auto"}),
                "visual_model": ([VISUAL_MODEL], {"default": VISUAL_MODEL}),
                "prompt_writer_model": (list(WRITER_MODELS), {"default": WRITER_MODELS[0]}),
                "short_direction": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "manual_prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "openrouter_api_key": ("STRING", {"default": "", "password": True}),
                "width": ("INT", {"default": 1280, "min": 1, "max": 8192}),
                "height": ("INT", {"default": 720, "min": 1, "max": 8192}),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.25, "max": 60.0, "step": 0.25}),
                "request_timeout": ("INT", {"default": DEFAULT_TIMEOUT, "min": 15, "max": 600}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("visual_analysis", "h3_prompt", "length")
    FUNCTION = "generate"
    CATEGORY = "MiniMax H3/Auto Prompt"

    def generate(
        self, start_image, prompt_mode, visual_model, prompt_writer_model,
        short_direction, manual_prompt, openrouter_api_key, width, height, duration,
        request_timeout=DEFAULT_TIMEOUT,
    ):
        length = _h3_length(duration)
        if prompt_mode == "manual":
            return _manual_result(manual_prompt, (length,))
        direction = (short_direction or "").strip()
        if not direction:
            raise ValueError("I2V auto mode requires a short_direction.")
        key = _resolve_key(openrouter_api_key)
        frames = _batch_frames(start_image)
        if not frames:
            raise ValueError("I2V auto mode requires a starting image.")
        analysis_content = [
            {"type": "text", "text": "USER DIRECTION (context only; not visual evidence):\n" + direction + "\n\nAnalyze the following actual starting image."},
            {"type": "image_url", "image_url": {"url": _jpeg_data_url(frames[0])}},
        ]
        analysis = _openrouter_completion(
            api_key=key, model=visual_model,
            messages=[
                {"role": "system", "content": I2V_ANALYZER_PROMPT},
                {"role": "user", "content": analysis_content},
            ],
            stage="I2V visual-analysis stage", timeout=request_timeout, max_tokens=4096,
        )
        writer_message = (
            "Write the final MiniMax H3 I2V prompt using the authoritative system instructions. "
            "The visual analysis is factual source-image context; the user direction controls "
            "the requested motion. Return only the final three-field prompt.\n\n"
            f"VISUAL ANALYSIS:\n{analysis}\n\nUSER DIRECTION:\n{direction}\n\n"
            f"TARGET FORMAT:\nWidth: {width}\nHeight: {height}\n"
            f"Aspect ratio: {_aspect(width, height)}\nDuration: {float(duration):.3f} seconds"
        )
        prompt = _openrouter_completion(
            api_key=key, model=prompt_writer_model,
            messages=[
                {"role": "system", "content": I2V_SYSTEM_PROMPT},
                {"role": "user", "content": writer_message},
            ],
            stage="I2V prompt-writing stage", timeout=request_timeout,
        )
        return (analysis, prompt, length)


class MiniMaxH3R2VTwoStagePrompt:
    VISUAL_INPUTS = tuple([f"image_{i}" for i in range(1, 10)] + [f"video_{i}" for i in range(1, 4)])
    AUDIO_INPUTS = tuple([f"video_audio_{i}" for i in range(1, 4)] + [f"audio_{i}" for i in range(1, 4)])

    @classmethod
    def INPUT_TYPES(cls):
        optional = {name: ("IMAGE", {"forceInput": True}) for name in cls.VISUAL_INPUTS}
        optional.update({name: ("AUDIO", {"forceInput": True}) for name in cls.AUDIO_INPUTS})
        return {
            "required": {
                "prompt_mode": (["auto", "manual"], {"default": "auto"}),
                "visual_model": ([VISUAL_MODEL], {"default": VISUAL_MODEL}),
                "prompt_writer_model": (list(WRITER_MODELS), {"default": WRITER_MODELS[0]}),
                "short_direction": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "manual_prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "sampled_video_frames": ("INT", {"default": 16, "min": 6, "max": 32, "step": 1}),
                "openrouter_api_key": ("STRING", {"default": "", "password": True}),
                "width": ("INT", {"default": 1280, "min": 1, "max": 8192}),
                "height": ("INT", {"default": 720, "min": 1, "max": 8192}),
                "duration": ("FLOAT", {"default": 8.0, "min": 0.25, "max": 60.0, "step": 0.25}),
                "job_type": (["auto", "standard", "replacement"], {"default": "auto"}),
                "request_timeout": ("INT", {"default": DEFAULT_TIMEOUT, "min": 15, "max": 600}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("visual_analysis", "h3_prompt", "debug")
    FUNCTION = "generate"
    CATEGORY = "MiniMax H3/Auto Prompt"

    @staticmethod
    def _media_manifest(media: dict[str, Any]) -> tuple[list[str], list[str]]:
        manifest: list[str] = []
        visual_tags: list[str] = []
        for slot in range(1, 10):
            if media.get(f"image_{slot}") is not None:
                tag = f"<Picture {slot}>"
                manifest.append(f"{tag}: standalone reference image")
                visual_tags.append(tag)
        audio_number = 0
        for slot in range(1, 4):
            video = media.get(f"video_{slot}")
            if video is None:
                continue
            tag = f"<Video {slot}>"
            audio_note = ""
            if media.get(f"video_audio_{slot}") is not None:
                audio_number += 1
                audio_note = f" <Audio {audio_number}>; soundtrack is preserved for MiniMax but was not sent to the Auto Prompt analyzer"
            manifest.append(f"{tag}{audio_note}")
            visual_tags.append(tag)
        for slot in range(1, 4):
            if media.get(f"audio_{slot}") is not None:
                audio_number += 1
                manifest.append(f"<Audio {audio_number}>: standalone audio preserved for MiniMax; not sent to the Auto Prompt analyzer")
        return manifest, visual_tags

    @staticmethod
    def _analysis_parts(direction: str, sampled_video_frames: int, media: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
        manifest, _ = MiniMaxH3R2VTwoStagePrompt._media_manifest(media)
        parts: list[dict[str, Any]] = [
            {"type": "text", "text": "REFERENCE MANIFEST:\n" + "\n".join(manifest) + "\n\nUSER DIRECTION (context only; not visual evidence):\n" + direction}
        ]
        for slot in range(1, 10):
            frames = _batch_frames(media.get(f"image_{slot}"))
            if not frames:
                continue
            parts.extend((
                {"type": "text", "text": f"REFERENCE IMAGE <Picture {slot}>"},
                {"type": "image_url", "image_url": {"url": _jpeg_data_url(frames[0])}},
            ))
        for slot in range(1, 4):
            frames = _batch_frames(media.get(f"video_{slot}"))
            if not frames:
                continue
            indices = _sample_indices(len(frames), sampled_video_frames)
            duration = len(frames) / VIDEO_FPS
            parts.append({"type": "text", "text": f"BEGIN <Video {slot}>: {len(indices)} sequential frames sampled evenly from the same {duration:.3f}s cropped/trimmed 24 fps clip that is supplied to MiniMax."})
            for sequence, frame_index in enumerate(indices, start=1):
                relative = 0.0 if len(frames) == 1 else frame_index / (len(frames) - 1)
                parts.extend((
                    {"type": "text", "text": f"<Video {slot}> Frame {sequence:02d}/{len(indices):02d}; chronological position {relative:.3f}; source frame {frame_index}."},
                    {"type": "image_url", "image_url": {"url": _jpeg_data_url(frames[frame_index])}},
                ))
            parts.append({"type": "text", "text": f"END <Video {slot}>"})
        return parts, manifest

    def generate(
        self, prompt_mode, visual_model, prompt_writer_model, short_direction,
        manual_prompt, sampled_video_frames, openrouter_api_key, width, height,
        duration, job_type, request_timeout=DEFAULT_TIMEOUT, **media,
    ):
        if prompt_mode == "manual":
            analysis, prompt = _manual_result(manual_prompt)
            return (analysis, prompt, "Manual mode: zero OpenRouter calls.")
        direction = (short_direction or "").strip()
        if not direction:
            raise ValueError("R2V auto mode requires a short_direction.")
        parts, manifest = self._analysis_parts(direction, sampled_video_frames, media)
        if not any(media.get(name) is not None for name in self.VISUAL_INPUTS):
            raise ValueError("R2V auto mode requires at least one reference image or video.")
        key = _resolve_key(openrouter_api_key)
        analysis = _openrouter_completion(
            api_key=key, model=visual_model,
            messages=[
                {"role": "system", "content": R2V_ANALYZER_PROMPT},
                {"role": "user", "content": parts},
            ],
            stage="R2V visual-analysis stage", timeout=request_timeout, max_tokens=6144,
        )
        try:
            authority = _r2v_writer_instructions(job_type)
        except Exception as exc:
            raise RuntimeError(f"R2V prompt-writing stage failed before API call: {exc}") from exc
        writer_message = (
            "The referenced media was analyzed upstream by Mistral. Write the final MiniMax H3 "
            "Ref2VA prompt from the factual analysis, direction, and exact manifest below. Do not "
            "rename, renumber, or invent reference tags. Audio references are available to MiniMax "
            "but were not analyzed; do not claim otherwise. Return only the final six-section "
            "Ref2VA prompt.\n\nREFERENCE MANIFEST:\n" + "\n".join(manifest)
            + f"\n\nJOB TYPE: {job_type}\n\nVISUAL/MOTION ANALYSIS:\n{analysis}\n\n"
            f"USER DIRECTION:\n{direction}\n\nTARGET FORMAT:\nWidth: {width}\nHeight: {height}\n"
            f"Aspect ratio: {_aspect(width, height)}\nDuration: {float(duration):.3f} seconds"
        )
        prompt = _openrouter_completion(
            api_key=key, model=prompt_writer_model,
            messages=[
                {"role": "system", "content": authority},
                {"role": "user", "content": writer_message},
            ],
            stage="R2V prompt-writing stage", timeout=request_timeout,
        )
        debug = (
            f"visual_model: {visual_model}\nwriter_model: {prompt_writer_model}\n"
            f"job_type: {job_type}\nsampled_video_frames: {sampled_video_frames}\n"
            "audio_analysis: disabled\nreference_manifest:\n" + "\n".join(manifest)
        )
        return (analysis, prompt, debug)


NODE_CLASS_MAPPINGS = {
    "MiniMaxOpenRouterKey": MiniMaxOpenRouterKey,
    "MiniMaxH3PromptMode": MiniMaxH3PromptMode,
    "MiniMaxH3I2VTwoStagePrompt": MiniMaxH3I2VTwoStagePrompt,
    "MiniMaxH3R2VTwoStagePrompt": MiniMaxH3R2VTwoStagePrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxOpenRouterKey": "MiniMax OpenRouter Key",
    "MiniMaxH3PromptMode": "MiniMax H3 Prompt Mode",
    "MiniMaxH3I2VTwoStagePrompt": "MiniMax H3 I2V Two-Stage Auto Prompt",
    "MiniMaxH3R2VTwoStagePrompt": "MiniMax H3 R2V Two-Stage Auto Prompt",
}
