import io
import os
import time
import json
import hashlib
import asyncio
import concurrent.futures
from typing import Tuple, Dict, List, Optional, Any

from PIL import Image
import torch
import numpy as np
import aiohttp


class NxdifyNode:
    """
    ComfyUI node for multi-image edit and image-to-video using Kie.ai Market APIs.

    Supports:
      - Seedream 4.5 Edit
      - Seedream 5.0 Lite Image to Image
      - Qwen2 Image Edit
      - Wan 2.7 Image Pro
      - Wan 2.7 Image to Video

    Uses 1-4 image inputs in ComfyUI, but individual endpoints may enforce stricter limits.
    Returns a ComfyUI IMAGE batch (BHWC), video URL string, and VIDEO output for video mode.
    """

    KIE_API_BASE = "https://api.kie.ai"
    KIE_UPLOAD_BASE = "https://kieai.redpandaai.co"
    CREATE_TASK_URL = f"{KIE_API_BASE}/api/v1/jobs/createTask"
    TASK_STATUS_URL = f"{KIE_API_BASE}/api/v1/jobs/recordInfo"
    FILE_UPLOAD_URL = f"{KIE_UPLOAD_BASE}/api/file-stream-upload"

    MODEL_SEEDREAM_45 = "seedream/4.5-edit"
    MODEL_SEEDREAM_5 = "seedream/5-lite-image-to-image"
    MODEL_QWEN2_IMAGE_EDIT = "qwen2/image-edit"
    MODEL_WAN_IMAGE_PRO = "wan/2-7-image-pro"
    MODEL_WAN_IMAGE_TO_VIDEO = "wan/2-7-image-to-video"

    GENERATION_TYPES = ["image", "video"]

    # Keep old/working selector name for backward compatibility.
    VERSION_OPTIONS = [
        "v5_lite",
        "v4.5",
        "qwen2_image_edit",
        "wan_2.7_image_pro",
    ]

    ASPECT_RATIOS = ["1:1", "4:3", "3:4", "16:9", "9:16", "2:3", "3:2", "21:9"]
    SEEDREAM_QUALITIES = ["basic", "high"]
    QWEN_OUTPUT_FORMATS = ["png", "jpeg"]

    WAN_ASPECT_RATIOS = ["auto", "1:1", "16:9", "4:3", "21:9", "3:4", "9:16", "8:1", "1:8"]
    WAN_RESOLUTIONS = ["1K", "2K", "4K"]

    VIDEO_MODES = ["first_frame", "first_and_last_frame"]
    VIDEO_RESOLUTIONS = ["720p", "1080p"]

    @classmethod
    def INPUT_TYPES(cls):
        # Keep the same core image inputs, then group the Kie.ai model controls.
        return {
            "required": {
                "face_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "kie_api_key": ("STRING", {"default": "", "password": True}),
                "generation_type": (cls.GENERATION_TYPES, {"default": "image"}),
                "seedream_version": (cls.VERSION_OPTIONS, {"default": "v4.5"}),
                "num_images": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1}),

                # Seedream 4.5 / 5.0 Lite
                "seedream_aspect_ratio": (cls.ASPECT_RATIOS, {"default": "1:1"}),
                "seedream_quality": (cls.SEEDREAM_QUALITIES, {"default": "basic"}),

                # Qwen2 Image Edit
                "qwen_image_size": (cls.ASPECT_RATIOS, {"default": "16:9"}),
                "qwen_output_format": (cls.QWEN_OUTPUT_FORMATS, {"default": "png"}),

                # Wan 2.7 Image Pro
                "wan_aspect_ratio": (cls.WAN_ASPECT_RATIOS, {"default": "auto"}),
                "wan_resolution": (cls.WAN_RESOLUTIONS, {"default": "2K"}),
                "wan_enable_sequential": ("BOOLEAN", {"default": False}),
                "wan_thinking_mode": ("BOOLEAN", {"default": False}),

                # Shared Kie.ai seed
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),

                # Wan 2.7 Image to Video
                "video_mode": (cls.VIDEO_MODES, {"default": "first_frame"}),
                "video_negative_prompt": (
                    "STRING",
                    {"multiline": True, "default": "blurry, flicker, low quality, distorted"},
                ),
                "video_resolution": (cls.VIDEO_RESOLUTIONS, {"default": "1080p"}),
                "video_duration": ("INT", {"default": 5, "min": 2, "max": 15, "step": 1}),
                "video_prompt_extend": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "body_image": ("IMAGE",),
                "breasts_image": ("IMAGE",),
                "dynamic_pose_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "VIDEO")
    RETURN_NAMES = ("image", "video_url", "video")
    FUNCTION = "execute"
    CATEGORY = "image/generation"

    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # Kie.ai image upload max is 10MB.
    MAX_CONCURRENT_DOWNLOADS = 8
    TASK_TIMEOUT_SECONDS = 15 * 60
    VIDEO_TASK_TIMEOUT_SECONDS = 30 * 60

    _image_url_cache: Dict[str, str] = {}

    def compress_image_bytes_max(self, image_bytes: bytes, max_bytes: int) -> bytes:
        if len(image_bytes) <= max_bytes:
            return image_bytes

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        base_w, base_h = img.size

        quality = 92
        scale = 1.0

        for _ in range(20):
            w = max(1, int(base_w * scale))
            h = max(1, int(base_h * scale))
            working = img if (w == base_w and h == base_h) else img.resize((w, h), Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            working.save(buf, format="JPEG", quality=quality, optimize=True)
            data = buf.getvalue()

            if len(data) <= max_bytes:
                return data

            if quality > 52:
                quality = max(52, quality - 10)
                continue

            if scale > 0.45:
                scale *= 0.85
                quality = 92
                continue

            return data

        return image_bytes

    def tensor_to_bytes(self, tensor: torch.Tensor) -> bytes:
        if len(tensor.shape) == 4:
            img_array = tensor[0].detach().cpu().numpy()
        else:
            img_array = tensor.detach().cpu().numpy()

        img_array = (np.clip(img_array, 0.0, 1.0) * 255.0).astype(np.uint8)

        if img_array.shape[2] == 4:
            alpha = img_array[:, :, 3:4].astype(np.float32) / 255.0
            rgb = img_array[:, :, :3].astype(np.float32)
            img_array = (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)
        elif img_array.shape[2] == 1:
            img_array = np.repeat(img_array, 3, axis=2)

        img = Image.fromarray(img_array)
        if img.mode != "RGB":
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95, optimize=True)
        return buf.getvalue()

    def pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_array = np.array(img).astype(np.float32) / 255.0
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=2)
            img_array = np.repeat(img_array, 3, axis=2)
        return torch.from_numpy(img_array)[None, ...]

    def _compute_image_hash(self, image_bytes: bytes) -> str:
        return hashlib.sha256(image_bytes).hexdigest()

    def _resolve_api_key(self, kie_api_key: str) -> str:
        """
        Prefer the workflow key if present; otherwise fall back to environment.
        Do not overwrite a valid env key with blank widget data.
        """
        key = (kie_api_key or "").strip()
        if not key:
            key = (os.getenv("KIE_API_KEY") or os.getenv("KIE_AI_API_KEY") or "").strip()

        if not key:
            raise ValueError("Kie.ai API key is required")

        return key

    def _auth_headers(self, api_key: str) -> Dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"}

    def _normalize_selector(self, seedream_version: str) -> str:
        version = seedream_version
        legacy_map = {
            "qwen_image_2_pro_edit": "qwen2_image_edit",
            "qwen_image_edit": "qwen2_image_edit",
        }
        if version in legacy_map:
            version = legacy_map[version]

        if version in {"nano_banana_pro", "flux", "flux_kontext"}:
            raise ValueError(f"Model '{seedream_version}' was removed from this Kie.ai version of the node.")

        if version not in self.VERSION_OPTIONS:
            raise ValueError(f"Unknown version/model: {seedream_version}")

        return version

    async def upload_ref_with_retry(
        self,
        image_bytes: bytes,
        api_key: str,
        use_cache: bool = True,
        max_attempts: int = 3,
    ) -> str:
        upload_start = time.time()
        original_size = len(image_bytes)

        image_hash = None
        if use_cache:
            image_hash = self._compute_image_hash(image_bytes)
            if image_hash in self._image_url_cache:
                print(f"[Nxdify] Cache hit (hash: {image_hash[:16]}...), skipping upload")
                return self._image_url_cache[image_hash]

        print(f"[Nxdify] Compressing image (original: {original_size} bytes)...")
        compressed = self.compress_image_bytes_max(image_bytes, self.MAX_IMAGE_SIZE)
        print(f"[Nxdify] Compressed to {len(compressed)} bytes")

        if image_hash is None:
            image_hash = self._compute_image_hash(image_bytes)

        file_name = f"nxdify-{image_hash[:16]}.jpg"
        headers = self._auth_headers(api_key)
        timeout = aiohttp.ClientTimeout(total=120)

        for attempt in range(max_attempts):
            try:
                print(f"[Nxdify] Uploading image to Kie.ai (attempt {attempt + 1}/{max_attempts})...")
                form = aiohttp.FormData()
                form.add_field("file", compressed, filename=file_name, content_type="image/jpeg")
                form.add_field("uploadPath", "images/nxdify")
                form.add_field("fileName", file_name)

                async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                    async with session.post(self.FILE_UPLOAD_URL, data=form) as resp:
                        payload = await resp.json(content_type=None)

                if payload.get("code") != 200 or payload.get("success") is False:
                    raise ValueError(f"Kie.ai upload failed: {payload}")

                data = payload.get("data") or {}
                url = data.get("downloadUrl") or data.get("fileUrl") or data.get("url")
                if not url:
                    raise ValueError(f"No upload URL found in Kie.ai response: {payload}")

                if use_cache:
                    self._image_url_cache[image_hash] = url

                print(f"[Nxdify] Upload successful in {time.time() - upload_start:.2f}s")
                return url

            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                err = str(e).lower()
                if "timeout" in err or "408" in err or "429" in err or "500" in err:
                    backoff = 2 + attempt * 3
                    print(f"[Nxdify] Upload issue; retry in {backoff}s: {e}")
                    await asyncio.sleep(backoff)
                    continue
                raise

        raise ValueError("Kie.ai upload failed after retries")

    async def _create_task(self, api_key: str, model: str, input_payload: dict) -> str:
        headers = self._auth_headers(api_key)
        headers["Content-Type"] = "application/json"
        payload = {"model": model, "input": input_payload}
        timeout = aiohttp.ClientTimeout(total=60)

        print(f"[Nxdify] Submitting Kie.ai task: {model}")
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            async with session.post(self.CREATE_TASK_URL, json=payload) as resp:
                result = await resp.json(content_type=None)

        if result.get("code") != 200:
            raise ValueError(f"Kie.ai createTask failed for {model}: {result}")

        data = result.get("data") or {}
        task_id = data.get("taskId")
        if not task_id:
            raise ValueError(f"Kie.ai createTask did not return taskId: {result}")

        return task_id

    async def _poll_task(self, api_key: str, task_id: str, timeout_seconds: int) -> dict:
        headers = self._auth_headers(api_key)
        timeout = aiohttp.ClientTimeout(total=60)
        start = time.time()
        delay = 2.0

        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            while time.time() - start < timeout_seconds:
                async with session.get(self.TASK_STATUS_URL, params={"taskId": task_id}) as resp:
                    payload = await resp.json(content_type=None)

                data = payload.get("data") or {}
                state = (data.get("state") or data.get("status") or "").lower()
                progress = data.get("progress")
                if progress is not None:
                    print(f"[Nxdify] Kie.ai task {task_id}: state={state or 'unknown'} progress={progress}")
                else:
                    print(f"[Nxdify] Kie.ai task {task_id}: state={state or 'unknown'}")

                if state == "success":
                    print(f"[Nxdify] Kie.ai task completed in {time.time() - start:.2f}s")
                    return data

                if state in {"fail", "failed", "error"}:
                    if self._extract_urls_from_result(data):
                        print(f"[Nxdify] Kie.ai task {task_id} failed but returned partial URL(s); keeping them.")
                        return data
                    fail_msg = data.get("failMsg") or data.get("error") or payload.get("msg") or "unknown error"
                    raise ValueError(f"Kie.ai task failed: {fail_msg}")

                await asyncio.sleep(delay)
                delay = min(delay * 1.25, 10.0)

        raise TimeoutError(f"Kie.ai task timed out after {timeout_seconds}s: {task_id}")

    async def _run_kie_task(
        self,
        api_key: str,
        model: str,
        input_payload: dict,
        timeout_seconds: int,
    ) -> dict:
        start = time.time()
        task_id = await self._create_task(api_key, model, input_payload)
        result = await self._poll_task(api_key, task_id, timeout_seconds)
        print(f"[Nxdify] Kie.ai job finished in {time.time() - start:.2f}s")
        return result

    def _extract_urls_from_result(self, result: Any) -> List[str]:
        urls: List[str] = []

        def add_url(value: Any) -> None:
            if isinstance(value, str) and value.startswith(("http://", "https://")) and value not in urls:
                urls.append(value)

        def walk(value: Any) -> None:
            if isinstance(value, str):
                stripped = value.strip()
                if stripped.startswith("{") or stripped.startswith("["):
                    try:
                        walk(json.loads(stripped))
                        return
                    except json.JSONDecodeError:
                        pass
                add_url(stripped)
                return

            if isinstance(value, list):
                for item in value:
                    walk(item)
                return

            if isinstance(value, dict):
                for key in (
                    "resultUrls",
                    "resultUrl",
                    "result_url",
                    "imageUrls",
                    "image_urls",
                    "videoUrls",
                    "video_urls",
                    "download_url",
                    "images",
                    "videos",
                    "urls",
                    "url",
                    "image_url",
                    "video_url",
                    "downloadUrl",
                    "fileUrl",
                ):
                    if key in value:
                        walk(value[key])

                for nested_key in ("result", "output", "data", "resultJson"):
                    if nested_key in value:
                        walk(value[nested_key])

        walk(result)
        return urls

    async def _download_one_image(self, session: aiohttp.ClientSession, url: str, idx: int) -> Image.Image:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise ValueError(f"Failed to download image {idx}: HTTP {resp.status}")
            b = await resp.read()
        return Image.open(io.BytesIO(b)).convert("RGB")

    async def _download_batch(self, urls: List[str]) -> torch.Tensor:
        connector = aiohttp.TCPConnector(limit=self.MAX_CONCURRENT_DOWNLOADS)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self._download_one_image(session, url, i) for i, url in enumerate(urls)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        pil_images = []
        failures = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                failures.append(f"download {idx + 1}: {result}")
                continue
            pil_images.append(result)

        if failures:
            print("[Nxdify] Some generated images failed to download:")
            for failure in failures:
                print(f"[Nxdify]   - {failure}")

        if not pil_images:
            raise ValueError("All generated image downloads failed.")

        tensors = [self.pil_to_tensor(img) for img in pil_images]
        return torch.cat(tensors, dim=0)

    def _video_from_file(self, video_path: str) -> Any:
        try:
            from comfy_api.latest import InputImpl
        except Exception as e:
            raise RuntimeError(
                "ComfyUI VIDEO output requires comfy_api.latest. Update ComfyUI or use video_url instead."
            ) from e

        return InputImpl.VideoFromFile(video_path)

    def _get_video_download_path(self, video_url: str) -> str:
        try:
            import folder_paths

            base_dir = folder_paths.get_temp_directory()
        except Exception:
            base_dir = os.path.join(os.getcwd(), "temp")

        video_dir = os.path.join(base_dir, "nxdify")
        os.makedirs(video_dir, exist_ok=True)
        digest = hashlib.sha256(video_url.encode("utf-8")).hexdigest()[:16]
        return os.path.join(video_dir, f"wan_video_{int(time.time())}_{digest}.mp4")

    async def _download_video_to_file(self, video_url: str) -> str:
        video_path = self._get_video_download_path(video_url)
        timeout = aiohttp.ClientTimeout(total=10 * 60)

        print(f"[Nxdify] Downloading video from Kie.ai: {video_url}")
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(video_url) as resp:
                if resp.status != 200:
                    raise ValueError(f"Failed to download video: HTTP {resp.status}")

                with open(video_path, "wb") as f:
                    while True:
                        chunk = await resp.content.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)

        print(f"[Nxdify] Video downloaded to: {video_path}")
        return video_path

    async def _run_repeated_image_tasks(
        self,
        api_key: str,
        model: str,
        input_payloads: List[dict],
    ) -> List[str]:
        tasks = [
            self._run_kie_task(api_key, model, input_payload, self.TASK_TIMEOUT_SECONDS)
            for input_payload in input_payloads
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        urls: List[str] = []
        failures = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                failures.append(f"task {idx + 1}: {result}")
                continue

            result_urls = self._extract_urls_from_result(result)
            if not result_urls:
                failures.append(f"task {idx + 1}: no image URLs returned")
                continue

            urls.extend(result_urls)

        if failures:
            print("[Nxdify] Some image generation tasks failed:")
            for failure in failures:
                print(f"[Nxdify]   - {failure}")

        if not urls:
            raise ValueError("All image generation tasks failed: " + "; ".join(failures))

        return urls

    async def generate_images_batch_tensor(
        self,
        api_key: str,
        seedream_version: str,
        image_urls: List[str],
        prompt: str,
        num_images: int,
        seedream_aspect_ratio: str,
        seedream_quality: str,
        qwen_image_size: str,
        qwen_output_format: str,
        wan_aspect_ratio: str,
        wan_resolution: str,
        wan_enable_sequential: bool,
        wan_thinking_mode: bool,
        seed: int,
    ) -> torch.Tensor:
        seedream_version = self._normalize_selector(seedream_version)

        if seedream_version == "v4.5":
            input_payload = {
                "prompt": prompt,
                "image_urls": image_urls,
                "aspect_ratio": seedream_aspect_ratio,
                "quality": seedream_quality,
                "nsfw_checker": False,
            }
            input_payloads = [dict(input_payload) for _ in range(num_images)]
            urls = await self._run_repeated_image_tasks(api_key, self.MODEL_SEEDREAM_45, input_payloads)

        elif seedream_version == "v5_lite":
            input_payload = {
                "prompt": prompt,
                "image_urls": image_urls,
                "aspect_ratio": seedream_aspect_ratio,
                "quality": seedream_quality,
                "nsfw_checker": False,
            }
            input_payloads = [dict(input_payload) for _ in range(num_images)]
            urls = await self._run_repeated_image_tasks(api_key, self.MODEL_SEEDREAM_5, input_payloads)

        elif seedream_version == "qwen2_image_edit":
            if not image_urls:
                raise ValueError("Qwen2 Image Edit requires one input image URL.")
            if len(image_urls) > 1:
                print("[Nxdify] Qwen2 Image Edit accepts one image_url; using the first uploaded image.")

            input_payloads = []
            for index in range(num_images):
                task_seed = min(seed + index, 2147483647)
                input_payloads.append(
                    {
                        "prompt": prompt,
                        "image_url": image_urls[0],
                        "image_size": qwen_image_size,
                        "output_format": qwen_output_format,
                        "seed": task_seed,
                        "nsfw_checker": False,
                    }
                )
            urls = await self._run_repeated_image_tasks(api_key, self.MODEL_QWEN2_IMAGE_EDIT, input_payloads)

        elif seedream_version == "wan_2.7_image_pro":
            if not wan_enable_sequential and num_images > 4:
                raise ValueError("Wan 2.7 Image Pro supports 1-4 images unless wan_enable_sequential is true.")

            if wan_thinking_mode and image_urls:
                print("[Nxdify] Wan thinking_mode is only available without input images; disabling for image edit.")

            input_payload = {
                "prompt": prompt,
                "input_urls": image_urls,
                "n": num_images,
                "enable_sequential": wan_enable_sequential,
                "resolution": wan_resolution,
                "thinking_mode": False if image_urls else wan_thinking_mode,
                "watermark": False,
                "seed": seed,
                "bbox_list": [[] for _ in image_urls],
                "nsfw_checker": False,
            }
            if wan_aspect_ratio != "auto":
                input_payload["aspect_ratio"] = wan_aspect_ratio

            result = await self._run_kie_task(
                api_key,
                self.MODEL_WAN_IMAGE_PRO,
                input_payload,
                self.TASK_TIMEOUT_SECONDS,
            )
            urls = self._extract_urls_from_result(result)

        else:
            raise ValueError(f"Unknown version/model: {seedream_version}")

        if not urls:
            raise ValueError("No image URLs found in Kie.ai result.")

        urls = urls[:num_images]
        print(f"[Nxdify] Kie.ai returned {len(urls)} image URL(s). Downloading...")

        batch = await self._download_batch(urls)
        print(f"[Nxdify] Returning batch tensor: shape={tuple(batch.shape)}")
        return batch

    async def generate_video_url(
        self,
        api_key: str,
        image_urls: List[str],
        prompt: str,
        video_mode: str,
        video_negative_prompt: str,
        video_resolution: str,
        video_duration: int,
        video_prompt_extend: bool,
        seed: int,
    ) -> str:
        if not image_urls:
            raise ValueError("Wan 2.7 Image to Video requires at least one input image.")
        if video_mode == "first_and_last_frame" and len(image_urls) < 2:
            raise ValueError("first_and_last_frame video mode requires two input images.")

        input_payload = {
            "prompt": prompt,
            "negative_prompt": video_negative_prompt,
            "first_frame_url": image_urls[0],
            "resolution": video_resolution,
            "duration": video_duration,
            "prompt_extend": video_prompt_extend,
            "watermark": False,
            "seed": seed,
            "nsfw_checker": False,
        }
        if video_mode == "first_and_last_frame":
            input_payload["last_frame_url"] = image_urls[1]

        result = await self._run_kie_task(
            api_key,
            self.MODEL_WAN_IMAGE_TO_VIDEO,
            input_payload,
            self.VIDEO_TASK_TIMEOUT_SECONDS,
        )
        urls = self._extract_urls_from_result(result)
        if not urls:
            raise ValueError("No video URL found in Kie.ai result.")

        print(f"[Nxdify] Returning video URL: {urls[0]}")
        return urls[0]

    async def process_async(
        self,
        face_image: torch.Tensor,
        prompt: str,
        kie_api_key: str,
        generation_type: str,
        seedream_version: str,
        num_images: int,
        seedream_aspect_ratio: str,
        seedream_quality: str,
        qwen_image_size: str,
        qwen_output_format: str,
        wan_aspect_ratio: str,
        wan_resolution: str,
        wan_enable_sequential: bool,
        wan_thinking_mode: bool,
        seed: int,
        video_mode: str,
        video_negative_prompt: str,
        video_resolution: str,
        video_duration: int,
        video_prompt_extend: bool,
        body_image: Optional[torch.Tensor] = None,
        breasts_image: Optional[torch.Tensor] = None,
        dynamic_pose_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, str, Any]:
        start = time.time()
        print("[Nxdify] ===== Starting process =====")

        if not prompt:
            raise ValueError("Prompt is required")

        api_key = self._resolve_api_key(kie_api_key)
        print(f"[Nxdify] Kie.ai API key present: {bool(api_key)} length={len(api_key)}")

        os.environ["KIE_API_KEY"] = api_key
        print("[Nxdify] Kie.ai key configured")

        if generation_type not in self.GENERATION_TYPES:
            raise ValueError(f"Unknown generation_type: {generation_type}")

        if generation_type == "video":
            provided: List[Tuple[str, torch.Tensor, bool]] = [("first_frame", face_image, True)]
            if video_mode == "first_and_last_frame":
                if body_image is None:
                    raise ValueError("first_and_last_frame video mode requires body_image as the last frame.")
                provided.append(("last_frame", body_image, True))
        else:
            provided = [("img1", face_image, True)]
            optional_images = [
                (body_image, True),
                (breasts_image, True),
                (dynamic_pose_image, False),
            ]
            for tens, use_cache in optional_images:
                if tens is not None:
                    provided.append((f"img{len(provided) + 1}", tens, use_cache))

        print("[Nxdify] Converting input tensors to bytes...")
        byte_items: List[Tuple[str, bytes, bool]] = []
        for label, tens, use_cache in provided:
            b = self.tensor_to_bytes(tens)
            byte_items.append((label, b, use_cache))

        print("[Nxdify] Provided images:", ", ".join([f"{label}={len(b)}B" for label, b, _ in byte_items]))

        print("[Nxdify] Uploading reference images to Kie.ai...")
        image_urls: List[str] = []
        for label, b, use_cache in byte_items:
            url = await self.upload_ref_with_retry(b, api_key=api_key, use_cache=use_cache)
            image_urls.append(url)
            print(f"[Nxdify] Uploaded {label} -> {url[:70]}...")

        if generation_type == "video":
            video_url = await self.generate_video_url(
                api_key=api_key,
                image_urls=image_urls,
                prompt=prompt,
                video_mode=video_mode,
                video_negative_prompt=video_negative_prompt,
                video_resolution=video_resolution,
                video_duration=video_duration,
                video_prompt_extend=video_prompt_extend,
                seed=seed,
            )
            video_path = await self._download_video_to_file(video_url)
            video = self._video_from_file(video_path)
            print(f"[Nxdify] ===== Total time: {time.time() - start:.2f}s =====")
            return face_image, video_url, video

        batch = await self.generate_images_batch_tensor(
            api_key=api_key,
            seedream_version=seedream_version,
            image_urls=image_urls,
            prompt=prompt,
            num_images=num_images,
            seedream_aspect_ratio=seedream_aspect_ratio,
            seedream_quality=seedream_quality,
            qwen_image_size=qwen_image_size,
            qwen_output_format=qwen_output_format,
            wan_aspect_ratio=wan_aspect_ratio,
            wan_resolution=wan_resolution,
            wan_enable_sequential=wan_enable_sequential,
            wan_thinking_mode=wan_thinking_mode,
            seed=seed,
        )

        print(f"[Nxdify] ===== Total time: {time.time() - start:.2f}s =====")
        return batch, "", None

    def _run_coroutine_in_new_loop(self, coro):
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            asyncio.set_event_loop(None)
            loop.close()

    def execute(
        self,
        face_image: torch.Tensor,
        prompt: str,
        kie_api_key: str,
        generation_type: str,
        seedream_version: str,
        num_images: int,
        seedream_aspect_ratio: str,
        seedream_quality: str,
        qwen_image_size: str,
        qwen_output_format: str,
        wan_aspect_ratio: str,
        wan_resolution: str,
        wan_enable_sequential: bool,
        wan_thinking_mode: bool,
        seed: int,
        video_mode: str,
        video_negative_prompt: str,
        video_resolution: str,
        video_duration: int,
        video_prompt_extend: bool,
        body_image: Optional[torch.Tensor] = None,
        breasts_image: Optional[torch.Tensor] = None,
        dynamic_pose_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, str, Any]:
        coro = self.process_async(
            face_image=face_image,
            prompt=prompt,
            kie_api_key=kie_api_key,
            generation_type=generation_type,
            seedream_version=seedream_version,
            num_images=num_images,
            seedream_aspect_ratio=seedream_aspect_ratio,
            seedream_quality=seedream_quality,
            qwen_image_size=qwen_image_size,
            qwen_output_format=qwen_output_format,
            wan_aspect_ratio=wan_aspect_ratio,
            wan_resolution=wan_resolution,
            wan_enable_sequential=wan_enable_sequential,
            wan_thinking_mode=wan_thinking_mode,
            seed=seed,
            video_mode=video_mode,
            video_negative_prompt=video_negative_prompt,
            video_resolution=video_resolution,
            video_duration=video_duration,
            video_prompt_extend=video_prompt_extend,
            body_image=body_image,
            breasts_image=breasts_image,
            dynamic_pose_image=dynamic_pose_image,
        )

        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                result = executor.submit(self._run_coroutine_in_new_loop, coro).result()
        except RuntimeError:
            result = self._run_coroutine_in_new_loop(coro)

        return result


NODE_CLASS_MAPPINGS = {"NxdifyNode": NxdifyNode}
NODE_DISPLAY_NAME_MAPPINGS = {"NxdifyNode": "Nxdify Kie.ai Generation"}
