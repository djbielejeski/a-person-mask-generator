import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2 as cv
import torch
import numpy as np
from PIL import Image
import mediapipe as mp
from .a_person_mask_generator_comfyui import APersonMaskGenerator
import folder_paths

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def get_face_landmarker_model_path() -> str:
    model_folder_name = "mediapipe"
    model_name = "face_landmarker.task"

    model_folder_path = os.path.join(folder_paths.models_dir, model_folder_name)
    model_file_path = os.path.join(model_folder_path, model_name)

    if not os.path.exists(model_file_path):
        import urllib.request

        model_url = f"https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/{model_name}"
        print(f"Downloading '{model_name}' model to '{model_file_path}'")
        os.makedirs(model_folder_path, exist_ok=True)
        try:
            urllib.request.urlretrieve(model_url, model_file_path)
        except Exception as e:
            if os.path.exists(model_file_path):
                os.remove(model_file_path)
            raise RuntimeError(
                f"Could not download '{model_name}' from {model_url}. "
                f"Download it manually and place it at '{model_file_path}'. ({e})"
            ) from e

    return model_file_path


class APersonFaceLandmarkMaskGenerator:
    # https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
    # order matters for these
    FACEMESH_FACE_OVAL = [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ]

    FACEMESH_LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
    FACEMESH_RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

    FACEMESH_LEFT_EYE = [
        362,
        382,
        381,
        380,
        374,
        373,
        390,
        249,
        263,
        466,
        388,
        387,
        386,
        385,
        384,
        398,
    ]
    FACEMESH_RIGHT_EYE = [
        33,
        7,
        163,
        144,
        145,
        153,
        154,
        155,
        133,
        173,
        157,
        158,
        159,
        160,
        161,
        246,
    ]

    FACEMESH_LEFT_PUPIL = [474, 475, 476, 477]
    FACEMESH_RIGHT_PUPIL = [469, 470, 471, 472]

    FACEMESH_LIPS = [
        61,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        291,
        308,
        324,
        318,
        402,
        317,
        14,
        87,
        178,
        88,
        95,
        185,
        40,
        39,
        37,
        0,
        267,
        269,
        270,
        409,
        415,
        310,
        311,
        312,
        13,
        82,
        81,
        42,
        183,
        78,
    ]

    def __init__(self):
        pass

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
                "face": false_widget,
                "left_eyebrow": false_widget,
                "right_eyebrow": false_widget,
                "left_eye": true_widget,
                "right_eye": true_widget,
                "left_pupil": false_widget,
                "right_pupil": false_widget,
                "lips": true_widget,
                "number_of_faces": (
                    "INT",
                    {"default": 1, "min": 1, "max": 10, "step": 1},
                ),
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

    @staticmethod
    def _fill(mask: np.ndarray, mesh_coords: list, indices: list):
        if max(indices) >= len(mesh_coords):
            return
        coords = [mesh_coords[p] for p in indices]
        cv.fillPoly(mask, [np.array(coords, dtype=np.int32)], 255)

    def draw_features(
        self, mask, mesh_coords, face, left_eyebrow, right_eyebrow,
        left_eye, right_eye, left_pupil, right_pupil, lips,
    ):
        if face:
            self._fill(mask, mesh_coords, self.FACEMESH_FACE_OVAL)
            return

        if left_eyebrow:
            self._fill(mask, mesh_coords, self.FACEMESH_LEFT_EYEBROW)
        if right_eyebrow:
            self._fill(mask, mesh_coords, self.FACEMESH_RIGHT_EYEBROW)
        if left_eye:
            self._fill(mask, mesh_coords, self.FACEMESH_LEFT_EYE)
        if right_eye:
            self._fill(mask, mesh_coords, self.FACEMESH_RIGHT_EYE)
        if left_pupil:
            self._fill(mask, mesh_coords, self.FACEMESH_LEFT_PUPIL)
        if right_pupil:
            self._fill(mask, mesh_coords, self.FACEMESH_RIGHT_PUPIL)
        if lips:
            self._fill(mask, mesh_coords, self.FACEMESH_LIPS)

    def generate_mask(
        self,
        images,
        face: bool,
        left_eyebrow: bool,
        right_eyebrow: bool,
        left_eye: bool,
        right_eye: bool,
        left_pupil: bool,
        right_pupil: bool,
        lips: bool,
        number_of_faces: int,
        confidence: float,
        refine_mask: bool,
    ):
        """Create a segmentation mask from an image

        Args:
            image (torch.Tensor): The image to create the mask for.
            face (bool): create a mask for the face.
            left_eyebrow (bool): create a mask for the left eyebrow.
            right_eyebrow (bool): create a mask for the right eyebrow.
            left_eye (bool): create a mask for the left eye.
            right_eye (bool): create a mask for the right eye.
            left_pupil (bool): create a mask for the left eye pupil.
            right_pupil (bool): create a mask for the right eye pupil.
            lips (bool): create a mask for the lips.

        Returns:
            torch.Tensor: The segmentation masks.
        """

        # use our APersonMaskGenerator with the face specified to get the image we should focus on

        a_person_mask_generator = None
        face_masks = None
        if refine_mask:
            a_person_mask_generator = APersonMaskGenerator()
            face_masks = a_person_mask_generator.get_mask_images(
                images=images,
                face_mask=True,
                background_mask=False,
                hair_mask=False,
                body_mask=False,
                clothes_mask=False,
                confidence=0.4,
                refine_mask=True,
            )

        any_feature_enabled = any([
            face, left_eyebrow, right_eyebrow, left_eye,
            right_eye, left_pupil, right_pupil, lips,
        ])

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=get_face_landmarker_model_path()),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=number_of_faces,
            min_face_detection_confidence=confidence,
            min_face_presence_confidence=confidence,
        )

        res_masks = []
        with FaceLandmarker.create_from_options(options) as face_landmarker:

            for index, image in enumerate(images):
                # Convert the Tensor to a PIL image
                i = 255.0 * image.cpu().numpy()
                image_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

                do_uncrop = False
                cropped_image_pil = image_pil

                # use the face mask to refine the image we are processing
                if refine_mask:
                    face_mask = face_masks[index]

                    # create the bounding box around the mask
                    bbox = a_person_mask_generator.get_bbox_for_mask(mask_image=face_mask)

                    if bbox != None:
                        cropped_image_pil = image_pil.crop(bbox)
                        do_uncrop = True

                # Process the image
                results = None
                if any_feature_enabled:
                    rgb = np.ascontiguousarray(
                        np.asarray(cropped_image_pil.convert("RGB"), dtype=np.uint8)
                    )
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    results = face_landmarker.detect(mp_image)

                img_width, img_height = cropped_image_pil.size
                mask = np.zeros((img_height, img_width), dtype=np.uint8)

                if results and results.face_landmarks:
                    for face_landmarks in results.face_landmarks:
                        mesh_coords = [
                            (int(point.x * img_width), int(point.y * img_height))
                            for point in face_landmarks
                        ]

                        self.draw_features(
                            mask=mask,
                            mesh_coords=mesh_coords,
                            face=face,
                            left_eyebrow=left_eyebrow,
                            right_eyebrow=right_eyebrow,
                            left_eye=left_eye,
                            right_eye=right_eye,
                            left_pupil=left_pupil,
                            right_pupil=right_pupil,
                            lips=lips,
                        )

                # Create the image
                mask_image = Image.fromarray(mask)

                if do_uncrop:
                    uncropped_mask_image = Image.new('RGBA', image_pil.size, (0, 0, 0))
                    uncropped_mask_image.paste(mask_image, bbox)
                    mask_image = uncropped_mask_image

                # convert PIL image to tensor image
                tensor_mask = mask_image.convert("RGB")
                tensor_mask = np.array(tensor_mask).astype(np.float32) / 255.0
                tensor_mask = torch.from_numpy(tensor_mask)[None,]
                tensor_mask = tensor_mask.squeeze(3)[..., 0]

                res_masks.append(tensor_mask)

        return (torch.cat(res_masks, dim=0),)
