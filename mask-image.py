import os
from functools import reduce
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MASK_OPTION_0_BACKGROUND = 'background'
MASK_OPTION_1_HAIR = 'hair'
MASK_OPTION_2_BODY = 'body (skin)'
MASK_OPTION_3_FACE = 'face (skin)'
MASK_OPTION_4_CLOTHES = 'clothes'

image_path = f'D:\\stable-diffusion\\Faces - IP Adaptor\\PXL_20230904_215912385.PORTRAIT.jpg'
output_image_path = f'D:\\stable-diffusion\\Faces - IP Adaptor\\PXL_20230904_215912385.PORTRAIT-masked.jpg'

if os.path.exists(output_image_path):
    os.remove(output_image_path)

image = Image.open(f"{image_path}")

mask_targets = [MASK_OPTION_0_BACKGROUND]
mask_dilation = 0

model_folder_path = 'mediapipe'
os.makedirs(model_folder_path, exist_ok=True)

model_path = os.path.join(model_folder_path, 'selfie_multiclass_256x256.tflite')
model_url = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite'
if not os.path.exists(model_path):
    import urllib.request
    print(f"Downloading 'selfie_multiclass_256x256.tflite' model")
    urllib.request.urlretrieve(model_url, model_path)

options = ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=model_path),
                                running_mode=VisionRunningMode.IMAGE,
                                output_category_mask=True)


def get_mediapipe_image(image: Image) -> mp.Image:
    # Convert gr.Image to NumPy array
    numpy_image = np.asarray(image)

    image_format = mp.ImageFormat.SRGB

    # Convert BGR to RGB (if necessary)
    if numpy_image.shape[-1] == 4:
        image_format = mp.ImageFormat.SRGBA
    elif numpy_image.shape[-1] == 3:
        image_format = mp.ImageFormat.SRGB
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)

    return mp.Image(image_format=image_format, data=numpy_image)

# Create the image segmenter
with ImageSegmenter.create_from_options(options) as segmenter:

    # Retrieve the masks for the segmented image
    media_pipe_image = get_mediapipe_image(image=image)
    segmented_masks = segmenter.segment(media_pipe_image)

    masks = []
    for i, target in enumerate(mask_targets):
        # https://developers.google.com/mediapipe/solutions/vision/image_segmenter#multiclass-model
        # 0 - background
        # 1 - hair
        # 2 - body - skin
        # 3 - face - skin
        # 4 - clothes
        # 5 - others(accessories)
        mask_index = 0
        if target == MASK_OPTION_1_HAIR:
            mask_index = 1
        if target == MASK_OPTION_2_BODY:
            mask_index = 2
        if target == MASK_OPTION_3_FACE:
            mask_index = 3
        if target == MASK_OPTION_4_CLOTHES:
            mask_index = 4

        masks.append(segmented_masks.confidence_masks[mask_index])

    image_data = media_pipe_image.numpy_view()

    # convert the image shape from "rgb" to "rgba" aka add the alpha channel
    if image_data.shape[-1] == 3:
        image_shape = (image_data.shape[0], image_data.shape[1], 4)
        alpha_channel = np.ones((image_shape[0], image_shape[1], 1), dtype=np.uint8) * 255
        image_data = np.concatenate((image_data, alpha_channel), axis=2)


    image_shape = image_data.shape

    mask_background_array = np.zeros(image_shape, dtype=np.uint8)
    mask_background_array[:] = (255, 255, 255, 255)

    #mask_foreground_array = np.zeros(image_shape, dtype=np.uint8)
    #mask_foreground_array[:] = (255, 255, 255, 255)

    mask_arrays = []
    for i, mask in enumerate(masks):
        condition = np.stack((mask.numpy_view(),) * image_shape[-1], axis=-1) > 0.25
        mask_array = np.where(condition, mask_background_array, image_data)
        mask_arrays.append(mask_array)

    # Merge our masks taking the maximum from each
    merged_mask_arrays = reduce(np.maximum, mask_arrays)

    # Dilate or erode the mask
    if mask_dilation > 0:
        merged_mask_arrays = cv2.dilate(merged_mask_arrays, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*mask_dilation + 1, 2*mask_dilation + 1), (mask_dilation, mask_dilation)))
    elif mask_dilation < 0:
        merged_mask_arrays = cv2.erode(merged_mask_arrays, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*mask_dilation + 1, 2*mask_dilation + 1), (mask_dilation, mask_dilation)))

    # Create the image
    mask_image = Image.fromarray(cv2.cvtColor(merged_mask_arrays, cv2.COLOR_BGR2RGB))

    mask_image.save(output_image_path)