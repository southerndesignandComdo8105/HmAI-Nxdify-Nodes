import io
import os
import time
import tempfile
import hashlib
import asyncio
import concurrent.futures
from typing import Tuple, Dict
from PIL import Image
import torch
import numpy as np
import fal_client as fal
from fal_client import client
import aiohttp


class NxdifyNode:
    """
    ComfyUI node for Nxdify image generation using FAL AI Seedream 4.5.
    Takes 4 reference images (Face, Body, Breasts, Dynamic Pose) and generates variations.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_image": ("IMAGE",),
                "body_image": ("IMAGE",),
                "breasts_image": ("IMAGE",),
                "dynamic_pose_image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "fal_api_key": ("STRING", {
                    "default": "",
                    "password": True
                }),
                "quality": (["auto_4K", "auto_2K"], {
                    "default": "auto_4K"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/generation"
    
    MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
    MAX_CONCURRENT = 8
    
    # Class-level cache for uploaded reference image URLs (hash -> URL)
    _image_url_cache: Dict[str, str] = {}
    
    def compress_image_bytes_max(self, image_bytes: bytes, max_bytes: int) -> bytes:
        """
        Compress image to fit under max_bytes.
        Strategy:
        1. Try reducing JPEG quality (start at 92, down to 52)
        2. If still too large, downscale image (start at 100%, down to 45%)
        3. Repeat until under limit or minimums reached
        """
        if len(image_bytes) <= max_bytes:
            return image_bytes
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        base_w, base_h = img.size
        
        quality = 92
        scale = 1.0
        
        for _ in range(20):  # Max 20 iterations
            w = max(1, int(base_w * scale))
            h = max(1, int(base_h * scale))
            
            # Resize if needed
            working = img if (w == base_w and h == base_h) else img.resize((w, h), Image.Resampling.LANCZOS)
            
            # Save as JPEG
            buf = io.BytesIO()
            working.save(buf, format="JPEG", quality=quality, optimize=True)
            data = buf.getvalue()
            
            if len(data) <= max_bytes:
                return data
            
            # Reduce quality first
            if quality > 52:
                quality = max(52, quality - 10)
                continue
            
            # Then downscale
            if scale > 0.45:
                scale = scale * 0.85
                quality = 92  # Reset quality
                continue
            
            # Can't compress further
            return data
        
        return image_bytes
    
    def tensor_to_bytes(self, tensor: torch.Tensor) -> bytes:
        """Convert ComfyUI image tensor to JPEG bytes."""
        # ComfyUI IMAGE tensors are in BHWC format (batch, height, width, channels)
        # Remove batch dimension to get HWC
        if len(tensor.shape) == 4:
            img_array = tensor[0].cpu().numpy()  # Shape: (height, width, channels)
        else:
            img_array = tensor.cpu().numpy()
        
        # Ensure values are in 0-255 range and convert to uint8
        img_array = (np.clip(img_array, 0.0, 1.0) * 255.0).astype(np.uint8)
        
        # Handle alpha channel if present
        if img_array.shape[2] == 4:
            # Convert RGBA to RGB with white background
            alpha = img_array[:, :, 3:4].astype(np.float32) / 255.0
            rgb = img_array[:, :, :3].astype(np.float32)
            img_array = (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)
        elif img_array.shape[2] == 1:
            # Handle grayscale - convert to RGB
            img_array = np.repeat(img_array, 3, axis=2)
        
        # Convert to PIL Image
        img = Image.fromarray(img_array)
        
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Save to bytes
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95, optimize=True)
        return buf.getvalue()
    
    def _compute_image_hash(self, image_bytes: bytes) -> str:
        """Compute SHA256 hash of image bytes for caching."""
        return hashlib.sha256(image_bytes).hexdigest()
    
    def _upload_file_sync(self, tmp_path: str) -> str:
        """Synchronous wrapper for upload_file to use with asyncio.to_thread."""
        return fal.upload_file(tmp_path)
    
    async def upload_ref_with_retry(self, image_bytes: bytes, use_cache: bool = True, max_attempts: int = 3) -> str:
        """Upload image with retry on timeout. Optionally use cache to avoid re-uploading."""
        upload_start = time.time()
        original_size = len(image_bytes)
        
        # Check cache first if enabled
        if use_cache:
            image_hash = self._compute_image_hash(image_bytes)
            if image_hash in self._image_url_cache:
                print(f"[Nxdify] Image found in cache (hash: {image_hash[:16]}...), skipping upload")
                return self._image_url_cache[image_hash]
        
        # Compress image first
        print(f"[Nxdify] Compressing image (original: {original_size} bytes)...")
        compressed = self.compress_image_bytes_max(image_bytes, self.MAX_IMAGE_SIZE)
        compression_ratio = (1 - len(compressed) / original_size) * 100 if original_size > 0 else 0
        print(f"[Nxdify] Compressed to {len(compressed)} bytes ({compression_ratio:.1f}% reduction)")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(compressed)
            tmp_path = tmp.name
        
        timeout_errors = []
        try:
            for attempt in range(max_attempts):
                try:
                    print(f"[Nxdify] Uploading image (attempt {attempt + 1}/{max_attempts})...")
                    attempt_start = time.time()
                    
                    # Upload to FAL (uses FAL_KEY environment variable set in process_async)
                    # upload_file expects a file path, not a BytesIO object
                    result = await asyncio.to_thread(self._upload_file_sync, tmp_path)
                    
                    attempt_elapsed = time.time() - attempt_start
                    print(f"[Nxdify] Upload completed in {attempt_elapsed:.2f} seconds")
                    
                    if isinstance(result, dict) and "url" in result:
                        url = result["url"]
                    elif isinstance(result, str):
                        url = result
                    else:
                        raise ValueError(f"Unexpected upload response: {result}")
                    
                    # Cache the URL if caching is enabled
                    if use_cache:
                        image_hash = self._compute_image_hash(image_bytes)
                        self._image_url_cache[image_hash] = url
                    
                    total_upload_time = time.time() - upload_start
                    print(f"[Nxdify] Image upload successful (total time: {total_upload_time:.2f} seconds)")
                    return url
                        
                except Exception as e:
                    # Check if this is a 408 Request Timeout error
                    error_str = str(e)
                    error_lower = error_str.lower()
                    is_408_timeout = (
                        "408" in error_str or
                        "request timeout" in error_lower or
                        "http/1.1 408" in error_lower or
                        "http 408" in error_lower
                    )
                    
                    # Check for other timeout errors
                    is_timeout = (
                        is_408_timeout or
                        "timeout" in error_lower or
                        isinstance(e, (TimeoutError, asyncio.TimeoutError)) or
                        (isinstance(e, aiohttp.ClientError) and "timeout" in error_lower)
                    )
                    
                    if is_408_timeout:
                        timeout_errors.append(f"Attempt {attempt + 1}: HTTP 408 Request Timeout")
                    
                    # If this is the last attempt and we had 408 timeouts, raise helpful exception
                    if attempt == max_attempts - 1:
                        if timeout_errors:
                            print(f"[Nxdify] Upload failed after {max_attempts} attempts")
                            raise RuntimeError(
                                f"Upload timed out after {max_attempts} attempts with HTTP 408 Request Timeout errors. "
                                f"The image may be too large. Please resize the image to a smaller resolution and try again. "
                                f"Errors: {'; '.join(timeout_errors)}"
                            )
                        print(f"[Nxdify] Upload failed on final attempt: {e}")
                        raise
                    
                    # If timeout error, retry with backoff
                    if is_timeout:
                        backoff = 2 + attempt * 3  # Exponential backoff: 2s, 5s, 8s
                        print(f"[Nxdify] Upload timeout error (attempt {attempt + 1}): {error_str[:100]}. Retrying in {backoff} seconds...")
                        await asyncio.sleep(backoff)
                        continue
                    
                    # Non-timeout error, fail immediately
                    print(f"[Nxdify] Upload failed with non-timeout error: {error_str[:100]}")
                    raise
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    
    def _subscribe_sync(self, endpoint: str, arguments: dict):
        """Subscribe to FAL API job synchronously (handles submit + polling internally)."""
        print(f"[Nxdify] Submitting job to FAL API: {endpoint}")
        start_time = time.time()
        result = fal.subscribe(endpoint, arguments=arguments, with_logs=False)
        elapsed = time.time() - start_time
        print(f"[Nxdify] FAL API job completed in {elapsed:.2f} seconds")
        return result
    
    async def generate_image(
        self,
        face_url: str,
        body_url: str,
        breasts_url: str,
        dynamic_pose_url: str,
        prompt: str,
        quality: str
    ) -> Image.Image:
        """Generate image using FAL AI Seedream 4.5 API."""
        print(f"[Nxdify] Starting image generation with quality: {quality}")
        print(f"[Nxdify] Reference images: face={face_url[:50]}..., body={body_url[:50]}..., breasts={breasts_url[:50]}..., pose={dynamic_pose_url[:50]}...")
        
        image_urls = [face_url, body_url, breasts_url, dynamic_pose_url]
        
        arguments = {
            "prompt": prompt,
            "image_size": quality,
            "num_images": 1,
            "max_images": 1,
            "enable_safety_checker": False,
            "image_urls": image_urls
        }
        
        print(f"[Nxdify] Calling FAL API subscribe (this will poll internally)...")
        subscribe_start = time.time()
        
        # Use subscribe which handles submit + polling internally
        result = await asyncio.to_thread(
            self._subscribe_sync,
            "fal-ai/bytedance/seedream/v4.5/edit",
            arguments
        )
        
        subscribe_elapsed = time.time() - subscribe_start
        print(f"[Nxdify] Subscribe call returned after {subscribe_elapsed:.2f} seconds")
        
        if not result:
            raise ValueError("No result returned from FAL AI API")
        
        print(f"[Nxdify] Processing result (type: {type(result).__name__})...")
        
        # Extract images from result (handle different response structures)
        images = None
        if isinstance(result, dict):
            if "images" in result:
                images = result["images"]
                print(f"[Nxdify] Found {len(images)} image(s) in result['images']")
            elif "output" in result and isinstance(result["output"], dict):
                images = result["output"].get("images")
                print(f"[Nxdify] Found {len(images) if images else 0} image(s) in result['output']['images']")
        
        if not images or len(images) == 0:
            raise ValueError("No images returned from FAL AI API")
        
        # Handle both dict and string image URLs
        if isinstance(images[0], dict):
            image_url = images[0].get("url") or images[0].get("image_url")
        else:
            image_url = images[0]
        
        if not image_url:
            raise ValueError("No image URL in result")
        
        print(f"[Nxdify] Image URL extracted: {image_url[:80]}...")
        print(f"[Nxdify] Downloading generated image...")
        download_start = time.time()
        
        # Download image
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status != 200:
                    raise ValueError(f"Failed to download image: HTTP {response.status}")
                image_bytes = await response.read()
        
        download_elapsed = time.time() - download_start
        print(f"[Nxdify] Image downloaded ({len(image_bytes)} bytes) in {download_elapsed:.2f} seconds")
        
        # Convert to PIL Image
        print(f"[Nxdify] Converting to PIL Image...")
        img = Image.open(io.BytesIO(image_bytes))
        final_img = img.convert("RGB")
        print(f"[Nxdify] Image generation complete. Final size: {final_img.size}")
        return final_img
    
    def pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL Image to ComfyUI tensor format (BHWC)."""
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Convert to numpy array (HWC format: height, width, channels)
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Ensure shape is (height, width, channels)
        if len(img_array.shape) == 2:
            # Grayscale - add channel dimension
            img_array = np.expand_dims(img_array, axis=2)
            img_array = np.repeat(img_array, 3, axis=2)  # Convert to RGB
        
        # Add batch dimension to get BHWC format: (batch, height, width, channels)
        tensor = torch.from_numpy(img_array)[None,]
        return tensor
    
    async def process_async(
        self,
        face_image: torch.Tensor,
        body_image: torch.Tensor,
        breasts_image: torch.Tensor,
        dynamic_pose_image: torch.Tensor,
        prompt: str,
        fal_api_key: str,
        quality: str
    ) -> torch.Tensor:
        """Async processing function."""
        process_start = time.time()
        print(f"[Nxdify] ===== Starting Nxdify image generation process =====")
        
        if not fal_api_key:
            raise ValueError("FAL API key is required")
        
        if not prompt:
            raise ValueError("Prompt is required")
        
        print(f"[Nxdify] Converting input tensors to bytes...")
        # Convert tensors to bytes
        face_bytes = self.tensor_to_bytes(face_image)
        body_bytes = self.tensor_to_bytes(body_image)
        breasts_bytes = self.tensor_to_bytes(breasts_image)
        dynamic_pose_bytes = self.tensor_to_bytes(dynamic_pose_image)
        print(f"[Nxdify] Image sizes: face={len(face_bytes)} bytes, body={len(body_bytes)} bytes, breasts={len(breasts_bytes)} bytes, pose={len(dynamic_pose_bytes)} bytes")
        
        # Set FAL API key as environment variable (FAL SDK reads from env)
        os.environ["FAL_KEY"] = fal_api_key
        print(f"[Nxdify] FAL API key configured")
        
        print(f"[Nxdify] Uploading reference images...")
        upload_start = time.time()
        
        # Upload fixed references (cached - only upload if changed)
        print(f"[Nxdify] Uploading face image (cached)...")
        face_url = await self.upload_ref_with_retry(face_bytes, use_cache=True)
        print(f"[Nxdify] Uploading body image (cached)...")
        body_url = await self.upload_ref_with_retry(body_bytes, use_cache=True)
        print(f"[Nxdify] Uploading breasts image (cached)...")
        breasts_url = await self.upload_ref_with_retry(breasts_bytes, use_cache=True)
        
        # Upload dynamic pose image (not cached - always upload)
        print(f"[Nxdify] Uploading dynamic pose image (not cached)...")
        dynamic_pose_url = await self.upload_ref_with_retry(dynamic_pose_bytes, use_cache=False)
        
        upload_elapsed = time.time() - upload_start
        print(f"[Nxdify] All images uploaded in {upload_elapsed:.2f} seconds")
        
        # Generate image
        print(f"[Nxdify] Starting image generation...")
        generation_start = time.time()
        generated_img = await self.generate_image(
            face_url,
            body_url,
            breasts_url,
            dynamic_pose_url,
            prompt,
            quality
        )
        generation_elapsed = time.time() - generation_start
        print(f"[Nxdify] Image generation completed in {generation_elapsed:.2f} seconds")
        
        # Convert to tensor
        print(f"[Nxdify] Converting PIL image to tensor...")
        result = self.pil_to_tensor(generated_img)
        
        total_elapsed = time.time() - process_start
        print(f"[Nxdify] ===== Total process time: {total_elapsed:.2f} seconds =====")
        return result
    
    def execute(
        self,
        face_image: torch.Tensor,
        body_image: torch.Tensor,
        breasts_image: torch.Tensor,
        dynamic_pose_image: torch.Tensor,
        prompt: str,
        fal_api_key: str,
        quality: str
    ) -> Tuple[torch.Tensor]:
        """Execute the node (synchronous wrapper for async processing)."""
        # Handle event loop - check if one exists
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we need to use a different approach
                # Create a new event loop in a thread
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.process_async(
                        face_image,
                        body_image,
                        breasts_image,
                        dynamic_pose_image,
                        prompt,
                        fal_api_key,
                        quality
                    ))
                    result = future.result()
            else:
                # Loop exists but not running, use it
                result = loop.run_until_complete(self.process_async(
                    face_image,
                    body_image,
                    breasts_image,
                    dynamic_pose_image,
                    prompt,
                    fal_api_key,
                    quality
                ))
        except RuntimeError:
            # No event loop, create one
            result = asyncio.run(self.process_async(
                face_image,
                body_image,
                breasts_image,
                dynamic_pose_image,
                prompt,
                fal_api_key,
                quality
            ))
        
        return (result,)


# Node export
NODE_CLASS_MAPPINGS = {
    "NxdifyNode": NxdifyNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NxdifyNode": "Nxdify Image Generation"
}

