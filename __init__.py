from .a_person_mask_generator_comfyui import APersonMaskGenerator

NODE_CLASS_MAPPINGS = {
    "APersonMaskGenerator": APersonMaskGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APersonMaskGenerator": "A Person Mask Generator"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']