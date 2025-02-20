import math
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from functools import reduce
import cv2
import torch
import numpy as np
from PIL import Image
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

import folder_paths


def get_a_person_mask_generator_model_path() -> str:
    model_folder_name = "mediapipe"
    model_name = "selfie_multiclass_256x256.tflite"

    model_folder_path = os.path.join(folder_paths.models_dir, model_folder_name)
    model_file_path = os.path.join(model_folder_path, model_name)

    if not os.path.exists(model_file_path):
        import wget

        model_url = f"https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/{model_name}"
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
        false_widget = (
            "BOOLEAN",
            {"default": False, "label_on": "enabled", "label_off": "disabled"},
        )
        true_widget = (
            "BOOLEAN",
            {"default": True, "label_on": "enabled", "label_off": "disabled"},
        )

        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "face_mask": true_widget,
                "background_mask": false_widget,
                "hair_mask": false_widget,
                "body_mask": false_widget,
                "clothes_mask": false_widget,
                "confidence": (
                    "FLOAT",
                    {"default": 0.40, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
                "refine_mask": true_widget,
            },
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

    def get_bbox_for_mask(self, mask_image: Image):
        # Convert the image to grayscale
        grayscale = mask_image.convert("L")

        # Create a binary mask where non-black pixels are white (255)
        mask_for_bbox = grayscale.point(lambda p: 255 if p > 0 else 0)

        # Get the bounding box of the non-black areas
        bbox = mask_for_bbox.getbbox()

        if bbox != None:
            left = bbox[0]
            upper = bbox[1]
            right = bbox[2]
            lower = bbox[3]

            bbox_width = right - left
            bbox_height = lower - upper

            # expand the box by 20% in each direction if possible
            bbox_padding_x = round(bbox_width * 0.2)
            bbox_padding_y = round(bbox_height * 0.2)

            # left, upper, right, lower
            bbox = (
                # left
                left - bbox_padding_x if left > bbox_padding_x else 0,
                # upper
                upper - bbox_padding_y if upper > bbox_padding_y else 0,
                # right
                right + bbox_padding_x if right < grayscale.width - bbox_padding_x else grayscale.width,
                # lower
                lower + bbox_padding_y if lower < grayscale.height - bbox_padding_y else grayscale.height,
            )

        return bbox

    def __get_mask(
            self,
            image: Image,
            segmenter,
            face_mask: bool,
            background_mask: bool,
            hair_mask: bool,
            body_mask: bool,
            clothes_mask: bool,
            confidence: float,
            refine_mask: bool,
    ) -> Image:
        # Retrieve the masks for the segmented image
        media_pipe_image = self.get_mediapipe_image(image=image)
        if any(
                [face_mask, background_mask, hair_mask, body_mask, clothes_mask]
        ):
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
                condition = (
                        np.stack((mask.numpy_view(),) * image_shape[-1], axis=-1)
                        > confidence
                )
                mask_array = np.where(
                    condition, mask_foreground_array, mask_background_array
                )
                mask_arrays.append(mask_array)

        # Merge our masks taking the maximum from each
        merged_mask_arrays = reduce(np.maximum, mask_arrays)

        # Create the image
        mask_image = Image.fromarray(merged_mask_arrays)

        # refine the mask by zooming in on the area where we detected our segments
        if refine_mask:
            bbox = self.get_bbox_for_mask(mask_image=mask_image)
            if bbox != None:
                cropped_image_pil = image.crop(bbox)

                cropped_mask_image = self.__get_mask(image=cropped_image_pil,
                                                   segmenter=segmenter,
                                                   face_mask=face_mask,
                                                   background_mask=background_mask,
                                                   hair_mask=hair_mask,
                                                   body_mask=body_mask,
                                                   clothes_mask=clothes_mask,
                                                   confidence=confidence,
                                                   refine_mask=False,
                                                   )

                updated_mask_image = Image.new('RGBA', image.size, (0, 0, 0))
                updated_mask_image.paste(cropped_mask_image, bbox)
                mask_image = updated_mask_image

        return mask_image

    def get_mask_images(
            self,
            images, # tensors
            face_mask: bool,
            background_mask: bool,
            hair_mask: bool,
            body_mask: bool,
            clothes_mask: bool,
            confidence: float,
            refine_mask: bool,
    ) -> list[Image]:
        a_person_mask_generator_model_path = get_a_person_mask_generator_model_path()
        a_person_mask_generator_model_buffer = None

        with open(a_person_mask_generator_model_path, "rb") as f:
            a_person_mask_generator_model_buffer = f.read()

        image_segmenter_base_options = BaseOptions(
            model_asset_buffer=a_person_mask_generator_model_buffer
        )
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=image_segmenter_base_options,
            running_mode=VisionRunningMode.IMAGE,
            output_category_mask=True,
        )

        mask_images: list[Image] = []

        # Create the image segmenter
        with ImageSegmenter.create_from_options(options) as segmenter:
            for tensor_image in images:
                # Convert the Tensor to a PIL image
                i = 255.0 * tensor_image.cpu().numpy()

                # The media pipe library does a much better job with images with an alpha channel for some reason.
                if i.shape[-1] == 3:  # If the image is RGB
                    # Add a fully transparent alpha channel (255)
                    i = np.dstack((i, np.full((i.shape[0], i.shape[1]), 255)))  # Create an RGBA image

                image_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                mask_image = self.__get_mask(
                    image=image_pil,
                    segmenter=segmenter,
                    face_mask=face_mask,
                    background_mask=background_mask,
                    hair_mask=hair_mask,
                    body_mask=body_mask,
                    clothes_mask=clothes_mask,
                    confidence=confidence,
                    refine_mask=refine_mask,
                )
                mask_images.append(mask_image)

        return mask_images

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
    ):
        """Create a segmentation mask from an image

        Args:
            image (torch.Tensor): The image to create the mask for.
            face_mask (bool): create a mask for the background.
            background_mask (bool): create a mask for the hair.
            hair_mask (bool): create a mask for the body .
            body_mask (bool): create a mask for the face.
            clothes_mask (bool): create a mask for the clothes.
            confidence (float): how confident the model is that the detected item is there.
            break_image_into_tiles ("none" or "auto"): break large images into tiles to improve detection.

        Returns:
            torch.Tensor: The segmentation masks.
        """

        mask_images = self.get_mask_images(
            images=images,
            face_mask=face_mask,
            background_mask=background_mask,
            hair_mask=hair_mask,
            body_mask=body_mask,
            clothes_mask=clothes_mask,
            confidence=confidence,
            refine_mask=refine_mask,
        )

        tensor_masks = []
        for mask_image in mask_images:
            # convert PIL image to tensor image
            tensor_mask = mask_image.convert("RGB")
            tensor_mask = np.array(tensor_mask).astype(np.float32) / 255.0
            tensor_mask = torch.from_numpy(tensor_mask)[None,]
            tensor_mask = tensor_mask.squeeze(3)[..., 0]

            tensor_masks.append(tensor_mask)

        return (torch.cat(tensor_masks, dim=0),)
