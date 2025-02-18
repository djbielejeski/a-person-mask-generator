from .a_person_mask_generator_comfyui import APersonMaskGenerator
from .a_person_face_landmark_mask_generator_comfyui import APersonFaceLandmarkMaskGenerator

NODE_CLASS_MAPPINGS = {
    "APersonMaskGenerator": APersonMaskGenerator,
    "APersonFaceLandmarkMaskGenerator": APersonFaceLandmarkMaskGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APersonMaskGenerator": "A Person Mask Generator",
    "APersonFaceLandmarkMaskGenerator": "A Person Face Landmark Mask Generator"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
