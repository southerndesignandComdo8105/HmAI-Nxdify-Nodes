# MiniMax H3 Two-Stage Auto Prompt

The I2V and R2V workflows use a two-stage OpenRouter pipeline:

1. `mistralai/mistral-medium-3-5` analyzes images.
2. `deepseek/deepseek-v4-flash` writes the final MiniMax H3 prompt from text.

`deepseek/deepseek-v4-pro` is also selectable as the writer. The code never sends
binary image data, image URLs, or video URLs to DeepSeek. It does not substitute a
different model or fall back to Gemini.

## Workflow Files

- `workflows/I2V_AutoPrompt_Integrated.json`: complete I2V generation workflow.
- `workflows/I2V_AutoPrompt_Module.json`: I2V prompt module for transplanting.
- `workflows/R2V_AutoPrompt_Integrated.json`: complete R2V generation workflow.
- `workflows/R2V_AutoPrompt_Module.json`: R2V ReferencePack and core module.

Both integrated workflows preserve their existing MiniMax model, text encoder,
VAE, LoRA, sampler, scheduler, resolution, duration, and save-video settings.

## I2V Auto Mode

`MiniMax H3 I2V Two-Stage Auto Prompt` accepts the actual starting `IMAGE`, a
short direction, target width and height, and duration.

Mistral returns factual visual analysis, keeping observed image content separate
from requested motion. DeepSeek receives that analysis as text, the short
direction, target format, duration, and the exact I2V prompt-writing instructions
recovered from the working RunPod reference. Its `h3_prompt` output is connected
to `MiniMaxH3ImageToVideo.prompt`.

The `length` output uses the original 24 fps MiniMax H3 frame-grid formula:

```text
max(5, round(duration_seconds * 24))
+ (5 - (max(5, round(duration_seconds * 24)) % 17)) % 17
```

## R2V Auto Mode

`MiniMaxH3ReferencePack` remains the only reference manager and media loader. Its
prompt provider is fixed to `none`, so it makes no language-model call. Its image,
video, soundtrack, and standalone-audio outputs remain connected to
`MiniMaxH3ReferenceToVideo`.

The same image and video tensors also reach `MiniMax H3 R2V Two-Stage Auto
Prompt`. This matters because ReferencePack has already applied the selected crop
and trim. Every video tensor is at 24 fps. The analyzer samples frames evenly over
that complete usable tensor, in temporal order, and labels them as sequential
frames belonging to `<Video N>`.

`sampled_video_frames` defaults to 16 and accepts 6 through 32. If a clip has
fewer frames than requested, every available frame is used once. Each frame is
JPEG-encoded for OpenRouter and downscaled only when its long edge exceeds 1536
pixels; crop, trim, ordering, and content are unchanged.

Standalone images retain `<Picture N>` tags. Videos retain `<Video N>` tags.
Soundtrack and standalone audio tags are reserved in the same order used by
MiniMaxRefPack. DeepSeek is told never to rename or renumber them.

The final writer instructions are read directly from the installed pinned
MiniMaxRefPack files:

- `minimax_refpack/system_prompt.md`
- `minimax_refpack/system_prompt_replacement.md`

`job_type` supports `standard`, `replacement`, and `auto`. Auto gives DeepSeek
both authoritative prompt registers and asks it to select the applicable one as
part of the second stage; it does not make a third classifier call.

## Audio Boundary

R2V audio sockets are preserved and continue to reach MiniMax. The Auto Prompt
path analyzes only:

- standalone reference images;
- sampled video frames;
- the user's short direction.

Mistral does not receive or analyze reference-video soundtracks or standalone
audio. DeepSeek receives explicit text noting which audio tags are reserved and
must not claim that audio was heard.

## Manual Mode

Set `prompt_mode` to `manual` and enter a complete `manual_prompt`. In manual
mode, both Python nodes return before key resolution, image encoding, prompt-file
loading, or the HTTP helper. They make zero Mistral, DeepSeek, and OpenRouter
calls. Manual R2V also works without references; manual I2V still expects the
starting image required by the MiniMax generation workflow.

## Inspection And Errors

Each workflow displays `Visual Analysis (Mistral)` separately from `Generated
Prompt`. R2V also displays an Auto Prompt debug summary with the selected models,
job type, frame-sample count, audio-analysis status, and exact reference manifest.
API keys are never printed.

Failures identify either the visual-analysis stage or prompt-writing stage and
include the selected model. There is no silent fallback. A missing
MiniMaxRefPack prompt asset is reported as an R2V prompt-writing error before the
DeepSeek call.

## Install

```bash
cd /ComfyUI/custom_nodes/HmAI-Nxdify-Nodes
COMFYUI_DIR=/ComfyUI bash scripts/install_minimax_autoprompt.sh
```

The installer installs the project requirements and pinned revisions of:

- `Hearmeman24/ComfyUI-MiniMaxRefPack` 0.3.5;
- `kijai/ComfyUI-KJNodes`;
- `rgthree/rgthree-comfy`;
- `Kosinkadink/ComfyUI-VideoHelperSuite`.

The project uses OpenRouter's HTTP API directly through `requests`; the generic
`ComfyUI-Openrouter_node` dependency is no longer needed.

## OpenRouter Key

Enter a runtime key in either two-stage node, or set an environment variable:

```bash
export OPENROUTER_API_KEY="your-runtime-key"
python3 main.py --listen
```

Blank key widgets resolve `OPENROUTER_API_KEY`, then `LLM_KEY`. Do not save a key
inside a workflow you plan to share.

## Model Assets

Auto Prompt uses hosted models and requires no local VLM. Download only missing
MiniMax generation assets with:

```bash
COMFYUI_DIR=/ComfyUI bash scripts/download_missing_minimax_models.sh
```

The downloader does not install personal or character LoRAs.

## Validation

Run the offline payload and graph tests with:

```bash
python3 -m unittest discover -s tests -p 'test_minimax_autoprompt_nodes.py' -v
```

The tests mock the HTTP boundary and verify image-to-Mistral payloads, full-span
video sampling, absence of `video_url`, text-only DeepSeek messages, exact model
IDs, tag retention, zero-call manual paths, valid workflow links, and final prompt
connections to the MiniMax core nodes.
