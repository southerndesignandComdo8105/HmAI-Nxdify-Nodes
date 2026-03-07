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
    ComfyUI node for multi-image edit using FAL endpoints.

    Supports:
      - Seedream 4.5 Edit
      - Seedream 5.0 Lite Edit
      - Nano Banana Pro Edit
      - Qwen Image 2 Pro Edit

    Uses up to 4 image inputs in ComfyUI, but individual endpoints may enforce stricter limits.
    Returns a ComfyUI IMAGE batch (BHWC).
    """

    ENDPOINT_SEEDREAM_45 = "fal-ai/bytedance/seedream/v4.5/edit"
    ENDPOINT_SEEDREAM_5 = "fal-ai/bytedance/seedream/v5/lite/edit"
    ENDPOINT_NANO_BANANA_PRO = "fal-ai/nano-banana-pro/edit"
    ENDPOINT_QWEN_IMAGE_2_PRO_EDIT = "fal-ai/qwen-image-2/pro/edit"

    SEEDREAM_IMAGE_SIZES = [
        "square_hd",
        "square",
        "portrait_4_3",
        "portrait_16_9",
        "landscape_4_3",
        "landscape_16_9",
        "auto_2K",
        "auto_3K",
        "auto_4K",
    ]

    SEEDREAM_45_VALID = {
        "square_hd",
        "square",
        "portrait_4_3",
        "portrait_16_9",
        "landscape_4_3",
        "landscape_16_9",
        "auto_2K",
        "auto_4K",
    }

    SEEDREAM_5_VALID = {
        "square_hd",
        "square",
        "portrait_4_3",
        "portrait_16_9",
        "landscape_4_3",
        "landscape_16_9",
        "auto_2K",
        "auto_3K",
    }

    QWEN_IMAGE_SIZES = [
        "square_hd",
        "square",
        "portrait_4_3",
        "portrait_16_9",
        "landscape_4_3",
        "landscape_16_9",
    ]

    NANO_RESOLUTIONS = ["1K", "2K", "4K"]
    NANO_ASPECT_RATIOS = ["auto", "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16"]
    NANO_OUTPUT_FORMATS = ["png", "jpeg", "webp"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    [
                        "seedream_v4_5",
                        "seedream_v5_lite",
                        "nano_banana_pro",
                        "qwen_image_2_pro_edit",
                    ],
                    {"default": "seedream_v5_lite"},
                ),
                "face_image": ("IMAGE",),
                "body_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "fal_api_key": ("STRING", {"default": "", "password": True}),
                "num_images": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),

                # Seedream
                "quality": (cls.SEEDREAM_IMAGE_SIZES, {"default": "auto_2K"}),

                # Nano Banana Pro
                "nano_resolution": (cls.NANO_RESOLUTIONS, {"default": "1K"}),
                "nano_aspect_ratio": (cls.NANO_ASPECT_RATIOS, {"default": "auto"}),
                "nano_output_format": (cls.NANO_OUTPUT_FORMATS, {"default": "png"}),

                # Qwen
                "qwen_image_size": (cls.QWEN_IMAGE_SIZES, {"default": "square_hd"}),
                "qwen_use_exact_2048": ("BOOLEAN", {"default": True}),
                "qwen_output_format": (["png", "jpeg", "webp"], {"default": "png"}),
            },
            "optional": {
                "breasts_image": ("IMAGE",),
                "dynamic_pose_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/generation"

    MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
    MAX_CONCURRENT_DOWNLOADS = 8

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

    def _upload_file_sync(self, tmp_path: str) -> Any:
        return fal.upload_file(tmp_path)

    async def upload_ref_with_retry(self, image_bytes: bytes, use_cache: bool = True, max_attempts: int = 3) -> str:
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

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(compressed)
            tmp_path = tmp.name

        try:
            for attempt in range(max_attempts):
                try:
                    print(f"[Nxdify] Uploading image (attempt {attempt + 1}/{max_attempts})...")
                    result = await asyncio.to_thread(self._upload_file_sync, tmp_path)

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

                    print(f"[Nxdify] Upload successful in {time.time() - upload_start:.2f}s")
                    return url

                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    err = str(e).lower()
                    if "timeout" in err or "408" in err:
                        backoff = 2 + attempt * 3
                        print(f"[Nxdify] Upload timeout; retry in {backoff}s")
                        await asyncio.sleep(backoff)
                        continue
                    raise
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _subscribe_sync(self, endpoint: str, arguments: dict):
        print(f"[Nxdify] Submitting job: {endpoint}")
        start = time.time()
        result = fal.subscribe(endpoint, arguments=arguments, with_logs=False)
        print(f"[Nxdify] FAL job completed in {time.time() - start:.2f}s")
        return result

    def _extract_image_urls_from_result(self, result: Any) -> List[str]:
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

    def _validate_seedream_quality(self, model: str, quality: str) -> None:
        if model == "seedream_v4_5" and quality not in self.SEEDREAM_45_VALID:
            raise ValueError(f"Quality '{quality}' is not valid for Seedream 4.5.")
        if model == "seedream_v5_lite" and quality not in self.SEEDREAM_5_VALID:
            raise ValueError(f"Quality '{quality}' is not valid for Seedream 5.0 Lite.")

    def _build_qwen_image_size(self, qwen_image_size: str, qwen_use_exact_2048: bool) -> Any:
        if qwen_use_exact_2048:
            return {"width": 2048, "height": 2048}
        return qwen_image_size

    async def _download_batch(self, urls: List[str]) -> torch.Tensor:
        connector = aiohttp.TCPConnector(limit=self.MAX_CONCURRENT_DOWNLOADS)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self._download_one_image(session, url, i) for i, url in enumerate(urls)]
            pil_images = await asyncio.gather(*tasks)

        tensors = [self.pil_to_tensor(img) for img in pil_images]
        return torch.cat(tensors, dim=0)

    async def generate_images_batch_tensor(
        self,
        model: str,
        image_urls: List[str],
        prompt: str,
        num_images: int,
        quality: str,
        nano_resolution: str,
        nano_aspect_ratio: str,
        nano_output_format: str,
        qwen_image_size: str,
        qwen_use_exact_2048: bool,
        qwen_output_format: str,
    ) -> torch.Tensor:
        if model == "seedream_v4_5":
            self._validate_seedream_quality(model, quality)
            endpoint = self.ENDPOINT_SEEDREAM_45
            arguments = {
                "prompt": prompt,
                "image_size": quality,
                "num_images": num_images,
                "max_images": 1,
                "enable_safety_checker": False,
                "image_urls": image_urls,
            }

        elif model == "seedream_v5_lite":
            self._validate_seedream_quality(model, quality)
            endpoint = self.ENDPOINT_SEEDREAM_5
            arguments = {
                "prompt": prompt,
                "image_size": quality,
                "num_images": num_images,
                "max_images": 1,
                "enable_safety_checker": False,
                "image_urls": image_urls,
            }

        elif model == "nano_banana_pro":
            endpoint = self.ENDPOINT_NANO_BANANA_PRO
            arguments = {
                "prompt": prompt,
                "image_urls": image_urls,
                "num_images": num_images,
                "resolution": nano_resolution,
                "aspect_ratio": nano_aspect_ratio,
                "output_format": nano_output_format,
                "safety_tolerance": "6",
            }

        elif model == "qwen_image_2_pro_edit":
            if not (1 <= len(image_urls) <= 3):
                raise ValueError(
                    f"Qwen Image 2 Pro Edit requires 1-3 input images; got {len(image_urls)}."
                )
            endpoint = self.ENDPOINT_QWEN_IMAGE_2_PRO_EDIT
            arguments = {
                "prompt": prompt,
                "image_urls": image_urls,
                "image_size": self._build_qwen_image_size(qwen_image_size, qwen_use_exact_2048),
                "enable_prompt_expansion": True,
                "enable_safety_checker": False,
                "num_images": num_images,
                "output_format": qwen_output_format,
            }

        else:
            raise ValueError(f"Unknown model: {model}")

        print(f"[Nxdify] Starting generation: model={model}, num_images={num_images}, refs={len(image_urls)}")
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

        batch = await self._download_batch(urls)
        print(f"[Nxdify] Returning batch tensor: shape={tuple(batch.shape)}")
        return batch

    async def process_async(
        self,
        model: str,
        face_image: torch.Tensor,
        body_image: torch.Tensor,
        prompt: str,
        fal_api_key: str,
        num_images: int,
        quality: str,
        nano_resolution: str,
        nano_aspect_ratio: str,
        nano_output_format: str,
        qwen_image_size: str,
        qwen_use_exact_2048: bool,
        qwen_output_format: str,
        breasts_image: Optional[torch.Tensor] = None,
        dynamic_pose_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        start = time.time()
        print("[Nxdify] ===== Starting process =====")

        if not fal_api_key:
            raise ValueError("FAL API key is required")
        if not prompt:
            raise ValueError("Prompt is required")

        os.environ["FAL_KEY"] = fal_api_key
        print("[Nxdify] FAL key configured")

        provided: List[Tuple[str, torch.Tensor, bool]] = [
            ("img1", face_image, True),
            ("img2", body_image, True),
        ]
        if breasts_image is not None:
            provided.append(("img3", breasts_image, True))
        if dynamic_pose_image is not None:
            provided.append(("img4", dynamic_pose_image, False))

        print("[Nxdify] Converting input tensors to bytes...")
        byte_items: List[Tuple[str, bytes, bool]] = []
        for label, tens, use_cache in provided:
            b = self.tensor_to_bytes(tens)
            byte_items.append((label, b, use_cache))

        print("[Nxdify] Provided images:", ", ".join([f"{label}={len(b)}B" for label, b, _ in byte_items]))

        print("[Nxdify] Uploading reference images...")
        image_urls: List[str] = []
        for label, b, use_cache in byte_items:
            url = await self.upload_ref_with_retry(b, use_cache=use_cache)
            image_urls.append(url)
            print(f"[Nxdify] Uploaded {label} -> {url[:70]}...")

        batch = await self.generate_images_batch_tensor(
            model=model,
            image_urls=image_urls,
            prompt=prompt,
            num_images=num_images,
            quality=quality,
            nano_resolution=nano_resolution,
            nano_aspect_ratio=nano_aspect_ratio,
            nano_output_format=nano_output_format,
            qwen_image_size=qwen_image_size,
            qwen_use_exact_2048=qwen_use_exact_2048,
            qwen_output_format=qwen_output_format,
        )

        print(f"[Nxdify] ===== Total time: {time.time() - start:.2f}s =====")
        return batch

    def execute(
        self,
        model: str,
        face_image: torch.Tensor,
        body_image: torch.Tensor,
        prompt: str,
        fal_api_key: str,
        num_images: int,
        quality: str,
        nano_resolution: str,
        nano_aspect_ratio: str,
        nano_output_format: str,
        qwen_image_size: str,
        qwen_use_exact_2048: bool,
        qwen_output_format: str,
        breasts_image: Optional[torch.Tensor] = None,
        dynamic_pose_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.process_async(
                            model=model,
                            face_image=face_image,
                            body_image=body_image,
                            breasts_image=breasts_image,
                            dynamic_pose_image=dynamic_pose_image,
                            prompt=prompt,
                            fal_api_key=fal_api_key,
                            num_images=num_images,
                            quality=quality,
                            nano_resolution=nano_resolution,
                            nano_aspect_ratio=nano_aspect_ratio,
                            nano_output_format=nano_output_format,
                            qwen_image_size=qwen_image_size,
                            qwen_use_exact_2048=qwen_use_exact_2048,
                            qwen_output_format=qwen_output_format,
                        ),
                    )
                    result = future.result()
            else:
                result = loop.run_until_complete(
                    self.process_async(
                        model=model,
                        face_image=face_image,
                        body_image=body_image,
                        breasts_image=breasts_image,
                        dynamic_pose_image=dynamic_pose_image,
                        prompt=prompt,
                        fal_api_key=fal_api_key,
                        num_images=num_images,
                        quality=quality,
                        nano_resolution=nano_resolution,
                        nano_aspect_ratio=nano_aspect_ratio,
                        nano_output_format=nano_output_format,
                        qwen_image_size=qwen_image_size,
                        qwen_use_exact_2048=qwen_use_exact_2048,
                        qwen_output_format=qwen_output_format,
                    )
                )
        except RuntimeError:
            result = asyncio.run(
                self.process_async(
                    model=model,
                    face_image=face_image,
                    body_image=body_image,
                    breasts_image=breasts_image,
                    dynamic_pose_image=dynamic_pose_image,
                    prompt=prompt,
                    fal_api_key=fal_api_key,
                    num_images=num_images,
                    quality=quality,
                    nano_resolution=nano_resolution,
                    nano_aspect_ratio=nano_aspect_ratio,
                    nano_output_format=nano_output_format,
                    qwen_image_size=qwen_image_size,
                    qwen_use_exact_2048=qwen_use_exact_2048,
                    qwen_output_format=qwen_output_format,
                )
            )

        return (result,)


NODE_CLASS_MAPPINGS = {"NxdifyNode": NxdifyNode}
NODE_DISPLAY_NAME_MAPPINGS = {"NxdifyNode": "Nxdify Multi-Image Edit"}
