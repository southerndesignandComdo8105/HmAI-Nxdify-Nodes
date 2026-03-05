import io
import os
import time
import tempfile
import hashlib
import asyncio
import concurrent.futures
from typing import Tuple, Dict, List, Optional, Any

from PIL import Image
import torch
import numpy as np
import fal_client as fal
import aiohttp


class NxdifyNode:
    """
    ComfyUI node for Nxdify image generation using FAL AI Seedream.
    Takes up to 4 reference images (Image1, Image2 required; Image3 optional; Image4 optional)
    and generates variations.

    Returns a ComfyUI IMAGE batch (BHWC) whose batch size == number of outputs returned.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_image": ("IMAGE",),  # treat as Image 1
                "body_image": ("IMAGE",),  # treat as Image 2
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "fal_api_key": ("STRING", {"default": "", "password": True}),

                # Choose which Seedream endpoint to call
                "seedream_version": (["v5_lite", "v4.5"], {"default": "v5_lite"}),

                # One unified dropdown (union); we validate at runtime based on seedream_version
                "quality": (
                    [
                        "square_hd",
                        "square",
                        "portrait_4_3",
                        "portrait_16_9",
                        "landscape_4_3",
                        "landscape_16_9",
                        "auto_2K",
                        "auto_3K",  # v5
                        "auto_4K",  # v4.5
                    ],
                    {"default": "auto_2K"},
                ),

                "num_images": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
            },
            "optional": {
                "breasts_image": ("IMAGE",),       # Image 3 (optional)
                "dynamic_pose_image": ("IMAGE",),  # Image 4 (optional)
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/generation"

    MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
    MAX_CONCURRENT_DOWNLOADS = 8

    # Class-level cache for uploaded reference image URLs (hash -> URL)
    _image_url_cache: Dict[str, str] = {}

    # -------------------------
    # Helpers: validation/config
    # -------------------------
    def _get_endpoint(self, seedream_version: str) -> str:
        if seedream_version == "v4.5":
            return "fal-ai/bytedance/seedream/v4.5/edit"
        # default v5 lite
        return "fal-ai/bytedance/seedream/v5/lite/edit"

    def _validate_quality(self, seedream_version: str, quality: str) -> None:
        allowed_v45 = {
            "square_hd",
            "square",
            "portrait_4_3",
            "portrait_16_9",
            "landscape_4_3",
            "landscape_16_9",
            "auto_2K",
            "auto_4K",
        }
        allowed_v5 = {
            "square_hd",
            "square",
            "portrait_4_3",
            "portrait_16_9",
            "landscape_4_3",
            "landscape_16_9",
            "auto_2K",
            "auto_3K",
        }

        allowed = allowed_v45 if seedream_version == "v4.5" else allowed_v5
        if quality not in allowed:
            raise ValueError(
                f"Quality '{quality}' is not valid for Seedream {seedream_version}. "
                f"Allowed: {sorted(allowed)}"
            )

    # -------------------------
    # Image conversion utilities
    # -------------------------
    def compress_image_bytes_max(self, image_bytes: bytes, max_bytes: int) -> bytes:
        """
        Compress image to fit under max_bytes.
        Strategy:
        1) Reduce JPEG quality (92 down to 52)
        2) If still too large, downscale (100% down to ~45%)
        """
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
                scale = scale * 0.85
                quality = 92
                continue

            return data

        return image_bytes

    def tensor_to_bytes(self, tensor: torch.Tensor) -> bytes:
        """Convert ComfyUI image tensor to JPEG bytes."""
        # ComfyUI IMAGE tensors are BHWC (batch, height, width, channels)
        if len(tensor.shape) == 4:
            img_array = tensor[0].detach().cpu().numpy()
        else:
            img_array = tensor.detach().cpu().numpy()

        img_array = (np.clip(img_array, 0.0, 1.0) * 255.0).astype(np.uint8)

        # Handle alpha / grayscale
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
        """Convert PIL Image to ComfyUI tensor format (BHWC) with batch dimension = 1."""
        if img.mode != "RGB":
            img = img.convert("RGB")

        img_array = np.array(img).astype(np.float32) / 255.0

        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=2)
            img_array = np.repeat(img_array, 3, axis=2)

        return torch.from_numpy(img_array)[None, ...]

    def _compute_image_hash(self, image_bytes: bytes) -> str:
        return hashlib.sha256(image_bytes).hexdigest()

    # -------------------------
    # FAL upload / subscribe
    # -------------------------
    def _upload_file_sync(self, tmp_path: str) -> Any:
        """Synchronous wrapper for fal.upload_file(path)."""
        return fal.upload_file(tmp_path)

    async def upload_ref_with_retry(self, image_bytes: bytes, use_cache: bool = True, max_attempts: int = 3) -> str:
        """Upload image with retry on timeout. Optionally use cache to avoid re-uploading."""
        upload_start = time.time()
        original_size = len(image_bytes)

        # Cache check
        image_hash = None
        if use_cache:
            image_hash = self._compute_image_hash(image_bytes)
            if image_hash in self._image_url_cache:
                print(f"[Nxdify] Cache hit (hash: {image_hash[:16]}...), skipping upload")
                return self._image_url_cache[image_hash]

        # Compress first
        print(f"[Nxdify] Compressing image (original: {original_size} bytes)...")
        compressed = self.compress_image_bytes_max(image_bytes, self.MAX_IMAGE_SIZE)
        compression_ratio = (1 - len(compressed) / original_size) * 100 if original_size > 0 else 0.0
        print(f"[Nxdify] Compressed to {len(compressed)} bytes ({compression_ratio:.1f}% reduction)")

        # Temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(compressed)
            tmp_path = tmp.name

        timeout_errors: List[str] = []
        try:
            for attempt in range(max_attempts):
                try:
                    print(f"[Nxdify] Uploading image (attempt {attempt + 1}/{max_attempts})...")
                    attempt_start = time.time()

                    result = await asyncio.to_thread(self._upload_file_sync, tmp_path)

                    attempt_elapsed = time.time() - attempt_start
                    print(f"[Nxdify] Upload completed in {attempt_elapsed:.2f}s")

                    if isinstance(result, dict) and "url" in result:
                        url = result["url"]
                    elif isinstance(result, str):
                        url = result
                    else:
                        raise ValueError(f"Unexpected upload response: {result}")

                    if use_cache:
                        if image_hash is None:
                            image_hash = self._compute_image_hash(image_bytes)
                        self._image_url_cache[image_hash] = url

                    total_upload_time = time.time() - upload_start
                    print(f"[Nxdify] Upload successful (total: {total_upload_time:.2f}s)")
                    return url

                except Exception as e:
                    error_str = str(e)
                    error_lower = error_str.lower()

                    is_408_timeout = (
                        "408" in error_str
                        or "request timeout" in error_lower
                        or "http/1.1 408" in error_lower
                        or "http 408" in error_lower
                    )
                    is_timeout = (
                        is_408_timeout
                        or "timeout" in error_lower
                        or isinstance(e, (TimeoutError, asyncio.TimeoutError))
                    )

                    if is_408_timeout:
                        timeout_errors.append(f"Attempt {attempt + 1}: HTTP 408 Request Timeout")

                    if attempt == max_attempts - 1:
                        if timeout_errors:
                            print(f"[Nxdify] Upload failed after {max_attempts} attempts")
                            raise RuntimeError(
                                f"Upload timed out after {max_attempts} attempts with HTTP 408 errors. "
                                f"Try resizing the image smaller. Errors: {'; '.join(timeout_errors)}"
                            )
                        print(f"[Nxdify] Upload failed on final attempt: {e}")
                        raise

                    if is_timeout:
                        backoff = 2 + attempt * 3  # 2s, 5s, 8s
                        print(f"[Nxdify] Upload timeout (attempt {attempt + 1}): {error_str[:120]}... retry in {backoff}s")
                        await asyncio.sleep(backoff)
                        continue

                    print(f"[Nxdify] Upload failed with non-timeout error: {error_str[:200]}")
                    raise

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _subscribe_sync(self, endpoint: str, arguments: dict):
        """Subscribe to FAL API job synchronously (submit + polling internally)."""
        print(f"[Nxdify] Submitting job: {endpoint}")
        start = time.time()
        result = fal.subscribe(endpoint, arguments=arguments, with_logs=False)
        elapsed = time.time() - start
        print(f"[Nxdify] FAL job completed in {elapsed:.2f}s")
        return result

    def _extract_image_urls_from_result(self, result: Any) -> List[str]:
        """Handle slightly different response structures and return list of URLs."""
        images = None
        if isinstance(result, dict):
            if "images" in result:
                images = result["images"]
            elif "output" in result and isinstance(result["output"], dict):
                images = result["output"].get("images")

        if not images:
            return []

        urls: List[str] = []
        for item in images:
            if isinstance(item, dict):
                url = item.get("url") or item.get("image_url")
            else:
                url = item
            if url:
                urls.append(url)
        return urls

    async def _download_one_image(self, session: aiohttp.ClientSession, url: str, idx: int) -> Image.Image:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise ValueError(f"Failed to download image {idx}: HTTP {resp.status}")
            b = await resp.read()
        return Image.open(io.BytesIO(b)).convert("RGB")

    # -------------------------
    # Generation
    # -------------------------
    async def generate_images_batch_tensor(
        self,
        seedream_version: str,
        image_urls: List[str],
        prompt: str,
        quality: str,
        num_images: int,
    ) -> torch.Tensor:
        """Generate images using selected Seedream endpoint and return a ComfyUI IMAGE batch tensor BHWC."""
        self._validate_quality(seedream_version, quality)

        endpoint = self._get_endpoint(seedream_version)
        print(
            f"[Nxdify] Starting generation: model={seedream_version}, endpoint={endpoint}, "
            f"image_size={quality}, num_images={num_images}, refs={len(image_urls)}"
        )

        arguments = {
            "prompt": prompt,
            "image_size": quality,
            "num_images": num_images,
            "max_images": 1,  # keeps output count <= num_images
            "enable_safety_checker": False,
            "image_urls": image_urls,
        }

        result = await asyncio.to_thread(self._subscribe_sync, endpoint, arguments)

        if not result:
            raise ValueError("No result returned from FAL API")

        urls = self._extract_image_urls_from_result(result)
        if not urls:
            if isinstance(result, dict):
                raise ValueError(f"No image URLs found in FAL result. Keys: {list(result.keys())}")
            raise ValueError(f"No image URLs found in FAL result. Type: {type(result)}")

        urls = urls[:num_images]
        print(f"[Nxdify] FAL returned {len(urls)} image URL(s). Downloading...")

        connector = aiohttp.TCPConnector(limit=self.MAX_CONCURRENT_DOWNLOADS)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self._download_one_image(session, url, i) for i, url in enumerate(urls)]
            pil_images = await asyncio.gather(*tasks)

        tensors = [self.pil_to_tensor(img) for img in pil_images]
        batch = torch.cat(tensors, dim=0)

        print(f"[Nxdify] Returning batch tensor: shape={tuple(batch.shape)}")
        return batch

    async def process_async(
        self,
        face_image: torch.Tensor,
        body_image: torch.Tensor,
        prompt: str,
        fal_api_key: str,
        seedream_version: str,
        quality: str,
        num_images: int,
        breasts_image: Optional[torch.Tensor] = None,
        dynamic_pose_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        process_start = time.time()
        print("[Nxdify] ===== Starting process =====")

        if not fal_api_key:
            raise ValueError("FAL API key is required")
        if not prompt:
            raise ValueError("Prompt is required")

        os.environ["FAL_KEY"] = fal_api_key
        print("[Nxdify] FAL key configured")

        # Build list of provided images in order: Image1, Image2, Image3?, Image4?
        provided: List[Tuple[str, torch.Tensor, bool]] = [
            ("img1", face_image, True),
            ("img2", body_image, True),
        ]
        if breasts_image is not None:
            provided.append(("img3", breasts_image, True))
        if dynamic_pose_image is not None:
            # keep your old "pose not cached" behavior
            provided.append(("img4", dynamic_pose_image, False))

        # Convert tensors to bytes
        print("[Nxdify] Converting input tensors to bytes...")
        byte_items: List[Tuple[str, bytes, bool]] = []
        for label, tens, use_cache in provided:
            b = self.tensor_to_bytes(tens)
            byte_items.append((label, b, use_cache))

        print("[Nxdify] Provided images:", ", ".join([f"{label}={len(b)}B" for label, b, _ in byte_items]))

        # Upload only what exists
        print("[Nxdify] Uploading reference images...")
        upload_start = time.time()

        image_urls: List[str] = []
        for label, b, use_cache in byte_items:
            url = await self.upload_ref_with_retry(b, use_cache=use_cache)
            image_urls.append(url)
            print(f"[Nxdify] Uploaded {label} -> {url[:70]}...")

        print(f"[Nxdify] Uploads done in {time.time() - upload_start:.2f}s")

        # Generate
        generation_start = time.time()
        batch = await self.generate_images_batch_tensor(
            seedream_version=seedream_version,
            image_urls=image_urls,
            prompt=prompt,
            quality=quality,
            num_images=num_images,
        )
        print(f"[Nxdify] Generation+download done in {time.time() - generation_start:.2f}s")
        print(f"[Nxdify] ===== Total time: {time.time() - process_start:.2f}s =====")
        return batch

    def execute(
        self,
        face_image: torch.Tensor,
        body_image: torch.Tensor,
        prompt: str,
        fal_api_key: str,
        seedream_version: str,
        quality: str,
        num_images: int,
        breasts_image: Optional[torch.Tensor] = None,
        dynamic_pose_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        """Synchronous wrapper for async processing."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.process_async(
                            face_image=face_image,
                            body_image=body_image,
                            prompt=prompt,
                            fal_api_key=fal_api_key,
                            seedream_version=seedream_version,
                            quality=quality,
                            num_images=num_images,
                            breasts_image=breasts_image,
                            dynamic_pose_image=dynamic_pose_image,
                        ),
                    )
                    result = future.result()
            else:
                result = loop.run_until_complete(
                    self.process_async(
                        face_image=face_image,
                        body_image=body_image,
                        prompt=prompt,
                        fal_api_key=fal_api_key,
                        seedream_version=seedream_version,
                        quality=quality,
                        num_images=num_images,
                        breasts_image=breasts_image,
                        dynamic_pose_image=dynamic_pose_image,
                    )
                )
        except RuntimeError:
            result = asyncio.run(
                self.process_async(
                    face_image=face_image,
                    body_image=body_image,
                    prompt=prompt,
                    fal_api_key=fal_api_key,
                    seedream_version=seedream_version,
                    quality=quality,
                    num_images=num_images,
                    breasts_image=breasts_image,
                    dynamic_pose_image=dynamic_pose_image,
                )
            )

        return (result,)


NODE_CLASS_MAPPINGS = {"NxdifyNode": NxdifyNode}
NODE_DISPLAY_NAME_MAPPINGS = {"NxdifyNode": "Nxdify Image Generation"}
