import os
import sys
import cv2
from typing import List, Optional, Tuple
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# MediaPipe (CPU in Python)
import mediapipe as mp
BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Comfy paths
import folder_paths


def get_a_person_mask_generator_model_path() -> str:
    model_folder_name = "mediapipe"
    model_name = "selfie_multiclass_256x256.tflite"

    model_folder_path = os.path.join(folder_paths.models_dir, model_folder_name)
    model_file_path = os.path.join(model_folder_path, model_name)

    if not os.path.exists(model_file_path):
        import urllib.request
        model_url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "image_segmenter/selfie_multiclass_256x256/float32/latest/"
            f"{model_name}"
        )
        print(f"Downloading '{model_name}' model")
        os.makedirs(model_folder_path, exist_ok=True)
        urllib.request.urlretrieve(model_url, model_file_path)

    return model_file_path


class APersonMaskGenerator:
    """
    Efficient + GPU-aware rewrite.
    - Keeps segmentation model loaded/reused
    - Moves only necessary data CPU<->GPU
    - Postprocessing done on GPU where available
    - Supports mini-batching to avoid GPU OOM
    """

    CATEGORY = "A Person Mask Generator - David Bielejeski"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "generate_mask"

    # -------- Node UI --------
    @classmethod
    def INPUT_TYPES(self):
        bool_off = ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"})
        bool_on  = ("BOOLEAN", {"default": True,  "label_on": "enabled", "label_off": "disabled"})

        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "face_mask": bool_on,
                "background_mask": bool_off,
                "hair_mask": bool_off,
                "body_mask": bool_off,
                "clothes_mask": bool_off,
                "confidence": ("FLOAT", {"default": 0.40, "min": 0.01, "max": 1.0, "step": 0.01}),
                "refine_mask": bool_on,
                # New performance options:
                "batch_size": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
                "compute_device": (  # where to do post-processing
                    ["auto", "cuda", "cpu"],
                ),
            },
        }

    def __init__(self):
        # Download once
        self._model_path = get_a_person_mask_generator_model_path()
        self._segmenter: Optional[ImageSegmenter] = None
        self._loaded_model_bytes: Optional[bytes] = None

    # -------- Segmenter lifecycle --------
    def _ensure_segmenter(self):
        if self._segmenter is not None:
            return

        if self._loaded_model_bytes is None:
            with open(self._model_path, "rb") as f:
                self._loaded_model_bytes = f.read()

        base_opts = BaseOptions(model_asset_buffer=self._loaded_model_bytes)
        options = ImageSegmenterOptions(
            base_options=base_opts,
            running_mode=VisionRunningMode.IMAGE,
            output_category_mask=True,  # we need per-class confidence masks
        )
        # Keep it open for the node lifetime
        self._segmenter = ImageSegmenter.create_from_options(options)

    def __del__(self):
        # Cleanly close the segmenter if it exists
        try:
            if self._segmenter is not None:
                self._segmenter.close()
                self._segmenter = None
        except Exception:
            pass

    # -------- Device helpers --------
    @staticmethod
    def _select_device(compute_device: str) -> torch.device:
        if compute_device == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if compute_device == "cpu":
            return torch.device("cpu")
        # auto
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- MediaPipe I/O adapters --------
    @staticmethod
    def _tensor_to_mp_image(image_t: torch.Tensor) -> mp.Image:
        """
        image_t: float tensor on CPU [H, W, C] in [0,1], C=3 or 4, RGB[A]
        Returns an mp.Image with SRGB/SRGBA format and uint8 data.
        """
        # Ensure CPU + contiguous
        if image_t.device.type != "cpu":
            image_t = image_t.cpu()
        image_t = image_t.contiguous()

        # Clamp + convert to uint8
        # If C==3, add alpha channel (as original implementation suggested better results)
        if image_t.shape[-1] == 3:
            # Add alpha 255
            H, W, _ = image_t.shape
            alpha = torch.ones((H, W, 1), dtype=image_t.dtype, device=image_t.device)
            image_t = torch.cat([image_t, alpha], dim=-1)

        np_img = (image_t * 255.0).clamp(0, 255).to(torch.uint8).numpy()

        if np_img.shape[-1] == 4:
            img_fmt = mp.ImageFormat.SRGBA
        else:
            img_fmt = mp.ImageFormat.SRGB  # (unlikely here; we force 4 above)

        return mp.Image(image_format=img_fmt, data=np_img)

    # -------- Mask combination helpers (GPU-accelerated) --------
    @staticmethod
    def _combine_selected_masks(
        class_masks_np: List[np.ndarray],
        selected_indices: List[int],
        confidence: float,
        out_device: torch.device,
    ) -> torch.Tensor:
        """
        class_masks_np: list of per-class float32 arrays in [0,1], shape [H,W]
        selected_indices: which classes to include
        Returns a single binary mask [H,W] float32 on out_device, values {0,1}.
        """
        if not selected_indices:
            # No selection means all background: return zeros
            # (Matches original behavior: if no masks selected it returned black)
            first = class_masks_np[0]
            return torch.zeros((first.shape[0], first.shape[1]), dtype=torch.float32, device=out_device)

        # Threshold each selected class, then logical OR across them.
        # Work on GPU
        masks = []
        for idx in selected_indices:
            m = torch.from_numpy(class_masks_np[idx])  # CPU float32
            m = (m > confidence)  # bool on CPU
            m = m.to(out_device)
            masks.append(m)

        merged_bool = masks[0]
        for m in masks[1:]:
            merged_bool = merged_bool | m

        return merged_bool.float()  # float binary mask {0,1} on out_device

    @staticmethod
    def _mask_bbox(mask: torch.Tensor) -> Optional[Tuple[int, int, int, int]]:
        """
        mask: [H,W] float (0/1) on any device
        Returns (left, top, right, bottom) or None if empty.
        """
        # Find nonzero coordinates
        nz = torch.nonzero(mask > 0.0, as_tuple=False)
        if nz.numel() == 0:
            return None
        ymin = int(nz[:, 0].min().item())
        ymax = int(nz[:, 0].max().item())
        xmin = int(nz[:, 1].min().item())
        xmax = int(nz[:, 1].max().item())

        # Expand bbox by 20% each direction (clamped later by caller with image size)
        h = ymax - ymin + 1
        w = xmax - xmin + 1
        pad_y = max(1, round(h * 0.2))
        pad_x = max(1, round(w * 0.2))
        return (xmin - pad_x, ymin - pad_y, xmax + pad_x + 1, ymax + pad_y + 1)  # right/bottom exclusive

    @staticmethod
    def _clamp_bbox(bbox: Tuple[int, int, int, int], H: int, W: int) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = bbox
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(W, x1)
        y1 = min(H, y1)
        if x1 <= x0 or y1 <= y0:
            return (0, 0, 0, 0)
        return (x0, y0, x1, y1)

    # -------- Single-image segmentation (called within batch loop) --------
    def _segment_one(
        self,
        image_t_cpu: torch.Tensor,  # [H,W,C] float on CPU
        class_selection: List[int],
        confidence: float,
        refine_mask: bool,
        out_device: torch.device,
    ) -> torch.Tensor:
        """
        Returns [H,W] float mask on out_device in [0,1]
        """
        # Convert to mp.Image
        mp_img = self._tensor_to_mp_image(image_t_cpu)
        seg = self._segmenter.segment(mp_img)

        # Map class indices:
        # 0 - background
        # 1 - hair
        # 2 - body - skin
        # 3 - face - skin
        # 4 - clothes
        # 5 - others(accessories)
        # MediaPipe returns a list of confidence masks aligned to input resolution
        class_masks_np = [cm.numpy_view().copy() for cm in seg.confidence_masks]  # ensure writable/owning

        base_mask = self._combine_selected_masks(
            class_masks_np=class_masks_np,
            selected_indices=class_selection,
            confidence=confidence,
            out_device=out_device,
        )

        if not refine_mask:
            return base_mask  # [H,W] float (0/1)

        # Refine: crop to bbox, re-segment, paste back
        H, W = base_mask.shape
        bbox0 = self._mask_bbox(base_mask)  # potentially on GPU
        if bbox0 is None:
            return base_mask

        bbox = self._clamp_bbox(bbox0, H, W)
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            return base_mask

        # Crop the original image (CPU tensor)
        cropped = image_t_cpu[y0:y1, x0:x1, :]

        # Re-run segmentation on the crop
        mp_img_crop = self._tensor_to_mp_image(cropped)
        seg2 = self._segmenter.segment(mp_img_crop)
        class_masks_np2 = [cm.numpy_view().copy() for cm in seg2.confidence_masks]

        refined_crop = self._combine_selected_masks(
            class_masks_np=class_masks_np2,
            selected_indices=class_selection,
            confidence=confidence,
            out_device=out_device,
        )

        # Paste refined crop back into a fresh empty mask
        refined_full = torch.zeros_like(base_mask, device=out_device)
        refined_full[y0:y1, x0:x1] = refined_crop
        return refined_full

    # -------- Public API required by Comfy --------
    def get_mask_images(
        self,
        images: torch.Tensor,  # list-like of image tensors or a batched tensor [B,H,W,C]
        face_mask: bool,
        background_mask: bool,
        hair_mask: bool,
        body_mask: bool,
        clothes_mask: bool,
        confidence: float,
        refine_mask: bool,
        batch_size: int,
        compute_device: str,
    ) -> List[torch.Tensor]:
        """
        Returns a list of [H,W] float masks on the selected device
        """
        self._ensure_segmenter()

        # Normalize input to [B,H,W,C] float CPU tensor
        # Comfy typically provides a list/tuple of tensors [H,W,C] or a single batched tensor.
        if isinstance(images, (list, tuple)):
            imgs = [img if isinstance(img, torch.Tensor) else torch.tensor(img) for img in images]
            # stack to batch if shapes equal; otherwise handle per-image
            same_shape = all(imgs[0].shape == im.shape for im in imgs)
            if same_shape:
                images_b = torch.stack(imgs, dim=0)
            else:
                # fall back to sequential handling with virtual batch
                images_b = imgs
        elif isinstance(images, torch.Tensor):
            images_b = images  # assume already [B,H,W,C] or [H,W,C]
            if images_b.dim() == 3:
                images_b = images_b.unsqueeze(0)
        else:
            raise TypeError("Unsupported images input type")

        # Mask class selection mapping:
        # indices in MediaPipe output list
        class_selection = []
        if background_mask: class_selection.append(0)
        if hair_mask:       class_selection.append(1)
        if body_mask:       class_selection.append(2)
        if face_mask:       class_selection.append(3)
        if clothes_mask:    class_selection.append(4)
        # (index 5 = accessories is intentionally not exposed to UI)

        out_device = self._select_device(compute_device)
        use_cuda = (out_device.type == "cuda")

        # Convert to CPU once for MediaPipe; keep a view for each
        # Expect images in [0,1] range float, shape [B,H,W,C], C in {3,4}
        def to_cpu_float(img_t: torch.Tensor) -> torch.Tensor:
            if img_t.dtype != torch.float32:
                img_t = img_t.float()
            if img_t.device.type != "cpu":
                img_t = img_t.cpu()
            return img_t

        # Create iterator over batches
        masks_out: List[torch.Tensor] = []

        if isinstance(images_b, list):
            # variable shapes – handle one by one in micro-batches of size 1
            iterable = [images_b[i:i+1] for i in range(len(images_b))]
        else:
            B = images_b.shape[0]
            iterable = [images_b[i:i+batch_size] for i in range(0, B, batch_size)]

        for batch in iterable:
            # batch is either a list length 1 with shape [H,W,C] or a tensor [b,H,W,C]
            if isinstance(batch, list):
                per_imgs = batch
            else:
                per_imgs = [batch[i] for i in range(batch.shape[0])]

            # Process each image in the batch (MediaPipe is single-image API)
            for img in per_imgs:
                img_cpu = to_cpu_float(img)
                mask_hw = self._segment_one(
                    image_t_cpu=img_cpu,
                    class_selection=class_selection,
                    confidence=confidence,
                    refine_mask=refine_mask,
                    out_device=out_device,
                )
                masks_out.append(mask_hw)

            # Free CUDA cache between mini-batches if requested
            if use_cuda:
                torch.cuda.empty_cache()

        return masks_out

    def generate_mask(
        self,
        images,
        face_mask: bool,
        background_mask: bool,
        hair_mask: bool,
        body_mask: bool,
        clothes_mask: bool,
        confidence: float,
        refine_mask: bool,
        batch_size: int = 8,
        compute_device: str = "auto",
    ):
        """
        Create segmentation masks for a batch of images.

        Args:
            images (torch.Tensor or List[torch.Tensor]): Input images [B,H,W,C] or list of [H,W,C],
                float in [0,1], RGB or RGBA.
            face_mask/background_mask/hair_mask/body_mask/clothes_mask (bool): which classes to include.
            confidence (float): threshold for class confidence.
            refine_mask (bool): if True, re-runs segmentation on a bbox crop for cleaner edges.
            batch_size (int): number of images to process per mini-batch (post-processing device aware).
            compute_device (str): "auto" | "cuda" | "cpu" for post-processing.

        Returns:
            (torch.Tensor,): A single tensor of shape [B,H,W] float in [0,1].
        """
        masks_list = self.get_mask_images(
            images=images,
            face_mask=face_mask,
            background_mask=background_mask,
            hair_mask=hair_mask,
            body_mask=body_mask,
            clothes_mask=clothes_mask,
            confidence=confidence,
            refine_mask=refine_mask,
            batch_size=batch_size,
            compute_device=compute_device,
        )

        # Stack into [B,H,W]
        # Keep on the chosen device to avoid needless transfers.
        if len(masks_list) == 0:
            # Shouldn't happen; return an empty tensor on CPU for safety
            return (torch.empty(0),)

        # If shapes vary, pad to max and stack (rare in Comfy workflows but safe)
        H_max = max(m.shape[0] for m in masks_list)
        W_max = max(m.shape[1] for m in masks_list)
        dev = masks_list[0].device

        padded = []
        for m in masks_list:
            H, W = m.shape
            if (H, W) == (H_max, W_max):
                padded.append(m)
            else:
                canvas = torch.zeros((H_max, W_max), dtype=m.dtype, device=dev)
                canvas[:H, :W] = m
                padded.append(canvas)

        masks_b = torch.stack(padded, dim=0)  # [B,H,W]
        return (masks_b,)
