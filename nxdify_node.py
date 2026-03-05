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
    ComfyUI node for Nxdify image generation using FAL AI Seedream 4.5.
    Takes 4 reference images (Face, Body, Breasts, Dynamic Pose) and generates variations.
    Returns a ComfyUI IMAGE batch (BHWC) whose batch size == number of outputs returned.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_image": ("IMAGE",),
                "body_image": ("IMAGE",),
                "breasts_image": ("IMAGE",),
                "dynamic_pose_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "fal_api_key": ("STRING", {"default": "", "password": True}),
                "quality": (["auto_4K", "auto_2K"], {"default": "auto_4K"}),

                # Requested number of outputs back (batch size). You said you want 4.
                "num_images": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/generation"

    MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
    MAX_CONCURRENT_DOWNLOADS = 8

    # Class-level cache for uploaded reference image URLs (hash -> URL)
    _image_url_cache: Dict[str, str] = {}

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

    async def generate_images_batch_tensor(
        self,
        face_url: str,
        body_url: str,
        breasts_url: str,
        dynamic_pose_url: str,
        prompt: str,
        quality: str,
        num_images: int,
    ) -> torch.Tensor:
        """Generate images using FAL Seedream 4.5 and return a ComfyUI IMAGE batch tensor BHWC."""
        print(f"[Nxdify] Starting generation: quality={quality}, num_images={num_images}")

        image_urls = [face_url, body_url, breasts_url, dynamic_pose_url]

        # Per the docs: total outputs can be num_images * max_images.
        # Keeping max_images=1 (as you stated) yields exactly num_images outputs, if the endpoint returns them.
        arguments = {
            "prompt": prompt,
            "image_size": quality,
            "num_images": num_images,
            "max_images": 1,
            "enable_safety_checker": False,
            "image_urls": image_urls,
        }

        result = await asyncio.to_thread(
            self._subscribe_sync,
            "fal-ai/bytedance/seedream/v4.5/edit",
            arguments,
        )

        if not result:
            raise ValueError("No result returned from FAL API")

        urls = self._extract_image_urls_from_result(result)
        if not urls:
            raise ValueError("No image URLs found in FAL result")

        # Keep at most requested count (endpoint might return more in some modes)
        urls = urls[:num_images]
        print(f"[Nxdify] FAL returned {len(urls)} image URL(s). Downloading...")

        # Download concurrently (bounded)
        connector = aiohttp.TCPConnector(limit=self.MAX_CONCURRENT_DOWNLOADS)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self._download_one_image(session, url, i) for i, url in enumerate(urls)]
            pil_images = await asyncio.gather(*tasks)

        # Convert each to tensor [1,H,W,C], then concat to [N,H,W,C]
        tensors = [self.pil_to_tensor(img) for img in pil_images]
        batch = torch.cat(tensors, dim=0)

        print(f"[Nxdify] Returning batch tensor: shape={tuple(batch.shape)}")
        return batch

    async def process_async(
        self,
        face_image: torch.Tensor,
        body_image: torch.Tensor,
        breasts_image: torch.Tensor,
        dynamic_pose_image: torch.Tensor,
        prompt: str,
        fal_api_key: str,
        quality: str,
        num_images: int,
    ) -> torch.Tensor:
        process_start = time.time()
        print(f"[Nxdify] ===== Starting process =====")

        if not fal_api_key:
            raise ValueError("FAL API key is required")
        if not prompt:
            raise ValueError("Prompt is required")

        # Convert tensors to jpeg bytes
        print(f"[Nxdify] Converting input tensors to bytes...")
        face_bytes = self.tensor_to_bytes(face_image)
        body_bytes = self.tensor_to_bytes(body_image)
        breasts_bytes = self.tensor_to_bytes(breasts_image)
        dynamic_pose_bytes = self.tensor_to_bytes(dynamic_pose_image)

        print(
            f"[Nxdify] Sizes: face={len(face_bytes)} body={len(body_bytes)} "
            f"breasts={len(breasts_bytes)} pose={len(dynamic_pose_bytes)}"
        )

        # Configure SDK
        os.environ["FAL_KEY"] = fal_api_key
        print(f"[Nxdify] FAL key configured")

        # Upload refs
        print(f"[Nxdify] Uploading reference images...")
        upload_start = time.time()

        face_url = await self.upload_ref_with_retry(face_bytes, use_cache=True)
        body_url = await self.upload_ref_with_retry(body_bytes, use_cache=True)
        breasts_url = await self.upload_ref_with_retry(breasts_bytes, use_cache=True)

        # dynamic pose not cached
        dynamic_pose_url = await self.upload_ref_with_retry(dynamic_pose_bytes, use_cache=False)

        upload_elapsed = time.time() - upload_start
        print(f"[Nxdify] Uploads done in {upload_elapsed:.2f}s")

        # Generate batch tensor
        generation_start = time.time()
        batch = await self.generate_images_batch_tensor(
            face_url=face_url,
            body_url=body_url,
            breasts_url=breasts_url,
            dynamic_pose_url=dynamic_pose_url,
            prompt=prompt,
            quality=quality,
            num_images=num_images,
        )
        generation_elapsed = time.time() - generation_start
        print(f"[Nxdify] Generation+download done in {generation_elapsed:.2f}s")

        total_elapsed = time.time() - process_start
        print(f"[Nxdify] ===== Total time: {total_elapsed:.2f}s =====")
        return batch

    def execute(
        self,
        face_image: torch.Tensor,
        body_image: torch.Tensor,
        breasts_image: torch.Tensor,
        dynamic_pose_image: torch.Tensor,
        prompt: str,
        fal_api_key: str,
        quality: str,
        num_images: int,
    ) -> Tuple[torch.Tensor]:
        """Synchronous wrapper for async processing."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.process_async(
                            face_image,
                            body_image,
                            breasts_image,
                            dynamic_pose_image,
                            prompt,
                            fal_api_key,
                            quality,
                            num_images,
                        ),
                    )
                    result = future.result()
            else:
                result = loop.run_until_complete(
                    self.process_async(
                        face_image,
                        body_image,
                        breasts_image,
                        dynamic_pose_image,
                        prompt,
                        fal_api_key,
                        quality,
                        num_images,
                    )
                )
        except RuntimeError:
            result = asyncio.run(
                self.process_async(
                    face_image,
                    body_image,
                    breasts_image,
                    dynamic_pose_image,
                    prompt,
                    fal_api_key,
                    quality,
                    num_images,
                )
            )

        return (result,)


NODE_CLASS_MAPPINGS = {"NxdifyNode": NxdifyNode}
NODE_DISPLAY_NAME_MAPPINGS = {"NxdifyNode": "Nxdify Image Generation"}

