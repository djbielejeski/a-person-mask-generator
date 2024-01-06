import os
import sys

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from functools import reduce
import wget
import cv2
import torch
import numpy as np
from PIL import Image
import mediapipe as mp

import folder_paths


def get_a_person_mask_generator_model_path() -> str:
    model_folder_name = 'mediapipe'
    model_name = 'selfie_multiclass_256x256.tflite'

    model_folder_path = os.path.join(folder_paths.models_dir, model_folder_name)
    model_file_path = os.path.join(model_folder_path, model_name)

    if not os.path.exists(model_file_path):
        model_url = f'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/{model_name}'
        print(f"Downloading '{model_name}' model")
        os.makedirs(model_folder_path, exist_ok=True)
        wget.download(model_url, model_file_path)

    return model_file_path


class APersonMaskGenerator:

    def __init__(self):
        # download the model if we need it
        get_a_person_mask_generator_model_path()

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required":
                {
                    "images": ("IMAGE",),
                },
            "optional":
                {
                    "face_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                    "background_mask": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "hair_mask": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "body_mask": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "clothes_mask": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "confidence": ("FLOAT", {"default": 0.40, "min": 0.01, "max": 1.0, "step": 0.01},),
                }
        }

    CATEGORY = "A Person Mask Generator - David Bielejeski"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)

    FUNCTION = "generate_mask"

    def get_mediapipe_image(self, image: Image) -> mp.Image:
        # Convert image to NumPy array
        numpy_image = np.asarray(image)

        image_format = mp.ImageFormat.SRGB

        # Convert BGR to RGB (if necessary)
        if numpy_image.shape[-1] == 4:
            image_format = mp.ImageFormat.SRGBA
        elif numpy_image.shape[-1] == 3:
            image_format = mp.ImageFormat.SRGB
            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)

        return mp.Image(image_format=image_format, data=numpy_image)

    def generate_mask(self, images, face_mask: bool, background_mask: bool, hair_mask: bool, body_mask: bool, clothes_mask: bool, confidence: float):

        """Create a segmentation mask from an image

        Args:
            image (torch.Tensor): The image to create the mask for.
            face_mask (bool): create a mask for the background.
            background_mask (bool): create a mask for the hair.
            hair_mask (bool): create a mask for the body .
            body_mask (bool): create a mask for the face.
            clothes_mask (bool): create a mask for the clothes.

        Returns:
            torch.Tensor: The segmentation masks.
        """

        a_person_mask_generator_model_path = get_a_person_mask_generator_model_path()
        a_person_mask_generator_model_buffer = None

        with open(a_person_mask_generator_model_path, "rb") as f:
            a_person_mask_generator_model_buffer = f.read()

        image_segmenter_base_options = mp.tasks.BaseOptions(model_asset_buffer=a_person_mask_generator_model_buffer)
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=image_segmenter_base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_category_mask=True)

        # Create the image segmenter
        res_masks = []
        with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
            for image in images:
                # Convert the Tensor to a PIL image
                i = 255. * image.cpu().numpy()
                image_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

                # create our foreground and background arrays for storing the mask results
                mask_background_array = np.zeros((image_pil.size[0], image_pil.size[1], 4), dtype=np.uint8)
                mask_background_array[:] = (0, 0, 0, 255)

                mask_foreground_array = np.zeros((image_pil.size[0], image_pil.size[1], 4), dtype=np.uint8)
                mask_foreground_array[:] = (255, 255, 255, 255)

                # Retrieve the masks for the segmented image
                media_pipe_image = self.get_mediapipe_image(image=image_pil)
                segmented_masks = segmenter.segment(media_pipe_image)

                # https://developers.google.com/mediapipe/solutions/vision/image_segmenter#multiclass-model
                # 0 - background
                # 1 - hair
                # 2 - body - skin
                # 3 - face - skin
                # 4 - clothes
                # 5 - others(accessories)
                masks = []
                if background_mask:
                    masks.append(segmented_masks.confidence_masks[0])
                if hair_mask:
                    masks.append(segmented_masks.confidence_masks[1])
                if body_mask:
                    masks.append(segmented_masks.confidence_masks[2])
                if face_mask:
                    masks.append(segmented_masks.confidence_masks[3])
                if clothes_mask:
                    masks.append(segmented_masks.confidence_masks[4])

                image_data = media_pipe_image.numpy_view()
                image_shape = image_data.shape

                # convert the image shape from "rgb" to "rgba" aka add the alpha channel
                if image_shape[-1] == 3:
                    image_shape = (image_shape[0], image_shape[1], 4)

                mask_background_array = np.zeros(image_shape, dtype=np.uint8)
                mask_background_array[:] = (0, 0, 0, 255)

                mask_foreground_array = np.zeros(image_shape, dtype=np.uint8)
                mask_foreground_array[:] = (255, 255, 255, 255)

                mask_arrays = []

                if len(masks) == 0:
                    mask_arrays.append(mask_background_array)
                else:
                    for i, mask in enumerate(masks):
                        condition = np.stack((mask.numpy_view(),) * image_shape[-1], axis=-1) > confidence
                        mask_array = np.where(condition, mask_foreground_array, mask_background_array)
                        mask_arrays.append(mask_array)

                # Merge our masks taking the maximum from each
                merged_mask_arrays = reduce(np.maximum, mask_arrays)

                # Create the image
                mask_image = Image.fromarray(merged_mask_arrays)

                # convert PIL image to tensor image
                tensor_mask = mask_image.convert("RGB")
                tensor_mask = np.array(tensor_mask).astype(np.float32) / 255.0
                tensor_mask = torch.from_numpy(tensor_mask)[None,]
                tensor_mask = tensor_mask.squeeze(3)[..., 0]

                res_masks.append(tensor_mask)

        return (torch.cat(res_masks, dim=0),)
