
# Imports
from PIL import Image, ImageFilter
import time

import sys

# GPT4-o
from openai import OpenAI
import base64
import requests
import json

# Grounded SAM2
import cv2
import supervision as sv
from supervision.draw.color import ColorPalette
from utils_groundedSAM2.supervision_utils import CUSTOM_COLOR_MAP
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Inpaint Anything
import torch
import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from lama_inpaint import inpaint_img_with_lama
from utils import dilate_mask

# WidowX
from interbotix_xs_modules.arm import InterbotixManipulatorXS
import interbotix_common_modules.angle_manipulation as ang
import rospy
from interbotix_xs_msgs.srv import MotorGains, MotorGainsRequest
import sys
import modern_robotics as mr
import time
import pyrealsense2 as rs  # camera


import os
import argparse
from scipy.ndimage import filters
from tqdm import tqdm
from transformers import TextStreamer
from absl import flags
import random
import pickle
import copy
import einops


# Import relevant libraries
from IPython import display
import jax
import tensorflow_datasets as tfds
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, overload

from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import default_kernel_init
from flax.linen.linear import DenseGeneral
from flax.linen.linear import DotGeneralT
from flax.linen.linear import PrecisionLike
from flax.linen.module import compact
from flax.linen.module import merge_param
from flax.linen.module import Module
from flax.linen.normalization import LayerNorm
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import random as rn
import os
from itertools import chain
import shutil
import gc
import psutil
import tqdm
import matplotlib
import tensorflow as tf
from absl import flags
import pickle
from octo.model.octo_model import OctoModel
from scipy.interpolate import UnivariateSpline



def warm_filter(img):
    """
    Standard warm filter for images.
    """
    increase_table = UnivariateSpline(x=[0, 64, 128, 255], y=[0, 95, 175, 255])(
        range(256)
    )

    middle_table = UnivariateSpline(x=[0, 64, 128, 255], y=[0, 75, 145, 255])(
        range(256)
    )
    decrease_table = UnivariateSpline(x=[0, 64, 128, 255], y=[0, 55, 105, 255])(
        range(256)
    )
    red_channel, green_channel, blue_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increase_table).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decrease_table).astype(np.uint8)

    filtered_img = cv2.merge((red_channel, green_channel, blue_channel))
    return filtered_img


def perturb_gaussian_noise(image, mask, std=0.25):
    """
    Input:
    image: numpy array of shape (H, W, 3)
    mask: numpy array of shape (H, W) - where to add noise

    Output:
    noised_image: numpy array of shape (H, W, 3)
    """

    # Convert the image to a float32 type
    image = image.astype(np.float32) / 255.0
    mask = mask.astype(np.float32) / 255.0

    # Define the Gaussian noise parameters
    mean = 0
    std_dev = std
    gaussian_noise = np.random.normal(mean, std_dev, image.shape)

    # Add the Gaussian noise to the image
    gaussian_noise[:, :, 0] = np.where(mask > 0, gaussian_noise[:, :, 0], 0)
    gaussian_noise[:, :, 1] = np.where(mask > 0, gaussian_noise[:, :, 1], 0)
    gaussian_noise[:, :, 2] = np.where(mask > 0, gaussian_noise[:, :, 2], 0)
    noisy_image = image + gaussian_noise
    # Clip the values to [0, 1] and convert back to [0, 255]
    noisy_image = np.clip(noisy_image, 0, 1)
    noisy_image = (noisy_image * 255).astype(np.uint8)

    return noisy_image


def perturb_gaussian_blur(image, mask, kernel_size=25):
    # Apply Gaussian blur to the whole image
    # The mask must be 0-255, not 0-1

    mask = mask * 255
    mask = mask.astype(np.uint8)

    image = Image.fromarray(image)
    mask = Image.fromarray(mask)

    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=kernel_size))

    # Composite the original image with the blurred image using the mask
    blurred_region = Image.composite(blurred_image, image, mask)
    blurred_region = np.asarray(blurred_region)
    return blurred_region


def encode_image(image_path):
    """
    Used for GPT4-o API
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def grounded_sam2_text(list):
    """
    Input:
    list: list of strings to track with Grounded SAM2
    Output:
    out: string of list elements in grounded SAM2 format
    """
    out = ". ".join(list)
    out += "."
    return out


def get_mask(object_name, class_names, detections):
    """
    Returns mask of object
    """
    obj_index = class_names.index(object_name)
    mask = detections.mask[obj_index]
    return mask


def grounded_sam2(img_path, text, save_annotations, save_directory):
    """
    Input:
    img_path: path to image
    text: objects to track with Grounded SAM2
	save_annotations: boolean to save output from Grounded-SAM2
	save_directory: where to save the Grounded-SAM2 output

    Output:
    detections
    """
    # environment settings
    # use bfloat16
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    image = Image.open(img_path)
    # sam2_predictor.set_image(image)
    sam2_predictor.set_image(np.array(image.convert("RGB")))
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.4,
        target_sizes=[image.size[::-1]],
    )
    # get the box prompt for SAM 2
    input_boxes = results[0]["boxes"].cpu().numpy()

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = results[0]["scores"].cpu().numpy().tolist()
    class_names = results[0]["labels"]
    class_ids = np.array(list(range(len(class_names))))

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(class_names, confidences)
    ]

    """
    Visualize image with supervision useful API
    """
    img = cv2.imread(img_path)
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids,  # (n, 4)  # (n, h, w)
    )

    if save_annotations:
        """
        Note that if you want to use default color map,
        you can set color=ColorPalette.DEFAULT
        """
        box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = box_annotator.annotate(
            scene=img.copy(), detections=detections
        )

        label_annotator = sv.LabelAnnotator(
            color=ColorPalette.from_hex(CUSTOM_COLOR_MAP)
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        cv2.imwrite(
            save_directory + "_groundingdino_annotated_image.jpg", annotated_frame
        )

        mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = mask_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        cv2.imwrite(
            save_directory + "_grounded_sam2_annotated_image_with_mask.jpg",
            annotated_frame,
        )

    return detections, class_names


def outpaint_anything(img, mask):
    """
    Input:
    img: numpy array of shape (H, W, 3)
    mask: numpy array of shape (H, W)

    Output:
    img: numpy array of shape (H, W, 3)
    """
    torch.autocast(device_type="cuda", dtype=torch.float32).__enter__()
    img_outpaint = inpaint_img_with_lama(
        img, mask, lama_config, lama_model, device=device
    )
    return img_outpaint


def gpt4o(img_path, language_instruction):
    """
	Few shot VLM and call to determine task-irrelevant objects in image
    Input:
    img_path: path to image for GPT4 to reason about
    language_instruction: instruction for robot to carry out

    Output: json file from GPT4 containing list of objects in image not relevant to task
    """

    api_key = "YOUR API KEY"

    client = OpenAI(
        organization="YOUR ORG",
        project="YOUR PROJECT",
        api_key=api_key,
    )

    # Read in examples for few-shot learning
    img_path1 = "path to img"
    img_path2 = "path to img"
    img_path3 = "path to img"
    img_path4 = "path to img"
    img_path5 = "path to img"

    fewshot_image1 = encode_image(img_path1)
    fewshot_image2 = encode_image(img_path2)
    fewshot_image3 = encode_image(img_path3)
    fewshot_image4 = encode_image(img_path4)
    fewshot_image5 = encode_image(img_path5)

    testtime_image = encode_image(img_path)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    # Create context and run
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an assistant helping a robot determine what objects in the image are relevant for completing its task.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You will be shown some text and images.",
                    },
                    {
                        "type": "text",
                        "text": "Example 1",
                    },
                    {
                        "type": "text",
                        "text": "Task: TASK1",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{fewshot_image1}"
                        },
                    },
                    {"type": "text", "text": '["obj1", "obj2"]'},
                    {
                        "type": "text",
                        "text": '["background1", "background2", "background3", "background4"]',
                    },
                    {
                        "type": "text",
                        "text": "Example 2",
                    },
                    {
                        "type": "text",
                        "text": "Task: TASK2",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{fewshot_image2}"
                        },
                    },
                    {"type": "text", "text": '["obj1", "obj2", "obj3"]'},
                    {"type": "text", "text": '["background1", "background2", "background3"]'},
                    {
                        "type": "text",
                        "text": "Example 3",
                    },
                    {
                        "type": "text",
                        "text": "Task: TASK3",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{fewshot_image3}"
                        },
                    },
                    {"type": "text", "text": '["obj1", "obj2"]'},
                    {
                        "type": "text",
                        "text": '["background1", "background2", "background3", "background4"]',
                    },
                    {
                        "type": "text",
                        "text": "Example 4",
                    },
                    {
                        "type": "text",
                        "text": "Task: TASK4",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{fewshot_image4}"
                        },
                    },
                    {"type": "text", "text": '["obj1", "obj2", "obj3"]'},
                    {"type": "text", "text": '["background1", "background2", "background3", "background4"]'},
                    {
                        "type": "text",
                        "text": "Example 5",
                    },
                    {
                        "type": "text",
                        "text": "Task: TASK5",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{fewshot_image5}"
                        },
                    },
                    {"type": "text", "text": '["obj1", "obj2", "obj3"]'},
                    {"type": "text", "text": '["background1", "background2"]'},
                    {
                        "type": "text",
                        "text": "The robotic arm in the image is given the following task: "
                        + language_instruction
                        + ". Provide a list of objects in the image that are not relevant for completing the task, called 'not_relevant_objects'. Then provide a list of backgrounds in the image that are not relevant for completing the task, called 'not_relevant_backgrounds'. Give your answer in the form of two different lists with one or two words per object. Respond in JSON file format only.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{testtime_image}"
                        },
                    },
                ],
            },
        ],
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    return response


def inpaint_random_color(img, mask):
    """
	Inpaints a region defined by a mask a random, neutral color
	
    Input:
    img: numpy array of shape (H, W, 3)
    mask: numpy array of shape (H, W)

    Output:
    img : numpy array of shape (H, W, 3)
    """
    base_value = np.random.randint(0, 255)

    # Generate the RGB color where R, G, and B are close (neutral)
    r = base_value + np.random.randint(-30, 30)
    g = base_value + np.random.randint(-30, 30)
    b = base_value + np.random.randint(-30, 30)

    # Make sure the values stay within the valid range (0-255)
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    # Convert the image to a float32 type
    noisy_image = img.copy()

    # Add the Gaussian noise to the image
    noisy_image[:, :, 0] = np.where(mask > 0, r, noisy_image[:, :, 0])
    noisy_image[:, :, 1] = np.where(mask > 0, g, noisy_image[:, :, 1])
    noisy_image[:, :, 2] = np.where(mask > 0, b, noisy_image[:, :, 2])

    return noisy_image, [r, g, b]


def inpaint_specific_color(img, mask, color):
    """
	Inpaints region of image defined by a mask a color given as (R,G,B)
    Input:
    img: numpy array of shape (H, W, 3)
    mask: numpy array of shape (H, W)
	color: list/np array

    Output:
    img : numpy array of shape (H, W, 3)
    """
    image = img.copy()
    r = color[0]
    g = color[1]
    b = color[2]

    # Add the Gaussian noise to the image
    image[:, :, 0] = np.where(mask > 0, r, img[:, :, 0])
    image[:, :, 1] = np.where(mask > 0, g, img[:, :, 1])
    image[:, :, 2] = np.where(mask > 0, b, img[:, :, 2])

    return image


def vlm_refine_output(response):
    """
    Input:
    response: json file from GPT4 containing list of objects and backgrounds in image not relevant to task

    Output:
    not_relevant_list: list of objects not relevant to task
    """
    # Parse the JSON
    parsed_response = json.loads(response.content)

    # Extract information
    all_objects = parsed_response["choices"][0]["message"]["content"]

    # Refine the JSON string
    all_objects_refine = all_objects.replace("json", "")
    all_objects_refine = all_objects_refine.replace("```", "")

    parsed_all_objects = json.loads(all_objects_refine)

    # Extract relevant and not relevant objects into separate lists
    not_relevant_objects_list = parsed_all_objects["not_relevant_objects"]
    not_relevant_backgrounds_list = parsed_all_objects["not_relevant_backgrounds"]

    return not_relevant_objects_list, not_relevant_backgrounds_list


def object_sensitivities(
    original_img,
    class_names_sensitivity,
    detections_sensitivity,
    w,
    thresh,
    N,
    language_instruction,
):
	"""
	Determines objects in image that the VLA is sensitive to
	"""
    num_objects = len(class_names_sensitivity)
    sensitivity = np.full((1, num_objects), True, dtype=bool)

    nonperturb_actions_octo = []  # holds all N samples
    peturb_actions_octo_all = (
        []
    )  # holds all N samples for k num_objects [K, N, n_steps, 7]

    delta_data = {}  # holds all deltas for k num_objects

    start_sensitivity = time.time()
    nonperturb_img = original_img.copy()
    nonperturb_img = warm_filter(nonperturb_img)
    nonperturb_img = cv2.resize(nonperturb_img, (256, 256))
    nonperturb_img = nonperturb_img[np.newaxis, np.newaxis, ...]
    nonperturb_img_curr = nonperturb_img.copy()

    texts = []
    timestep_pad_mask = []
    for n in range(N):
        if n == 0:
            nonperturb_img_curr = nonperturb_img_curr
            texts.append(language_instruction)
            timestep_pad_mask.append([True])
        else:
            nonperturb_img_curr = np.vstack([nonperturb_img_curr, nonperturb_img])
            texts.append(language_instruction)
            timestep_pad_mask.append([True])

    nonperturb_observation = {
        "image_primary": nonperturb_img_curr,
        "timestep_pad_mask": np.array(timestep_pad_mask),
    }

    task = model.create_tasks(texts=texts)

    nonperturb_actions_octo = model.sample_actions(
        nonperturb_observation,
        task,
        unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"],
        rng=jax.random.PRNGKey(0),
    )
    # Turn into array
    nonperturb_actions_octo = np.array(nonperturb_actions_octo)  # [N, n_steps, 7]

    # Determine which objects octo is sensitve to
    perturb_images = []
    perturb_actions_all_items_curr = []
    for i in range(num_objects):
        mask = get_mask(
            class_names_sensitivity[i],
            class_names_sensitivity,
            detections_sensitivity,
        )
        dilate_size = 0
        mask_dilate = dilate_mask(mask, dilate_size)

        texts = []
        timestep_pad_mask = []
        for n in range(N):
            if n == 0:
                random_kernel_size = np.random.randint(15, 30)

                # Perturb
                perturb_img_curr = perturb_gaussian_blur(
                    original_img, mask_dilate, kernel_size=random_kernel_size
                )
      
                perturb_img_curr = warm_filter(perturb_img_curr)

                # Save Image for Analysis
                perturb_images.append(perturb_img_curr)
                

                perturb_img_curr = cv2.resize(perturb_img_curr, (256, 256))
                perturb_img_curr = perturb_img_curr[np.newaxis, np.newaxis, ...]
                texts.append(language_instruction)
                timestep_pad_mask.append([True])
            else:
                random_kernel_size = np.random.randint(15, 30)

                # Perturb
                perturb_img = perturb_gaussian_blur(
                    original_img, mask_dilate, kernel_size=random_kernel_size
                )
                # perturb_img = perturb_gaussian_noise(original_img, mask_dilate)
                perturb_img = warm_filter(perturb_img)

                # Save Image for Analysis
                perturb_images.append(perturb_img)
               

                perturb_img = cv2.resize(perturb_img, (256, 256))
                perturb_img = perturb_img[np.newaxis, np.newaxis, ...]

                perturb_img_curr = np.vstack([perturb_img_curr, perturb_img])
                texts.append(language_instruction)
                timestep_pad_mask.append([True])

        perturb_observation = {
            "image_primary": perturb_img_curr,
            "timestep_pad_mask": np.array(timestep_pad_mask),
        }
        task = model.create_tasks(texts=texts)

        perturb_actions_octo_curr = []  # holds all N samples for current object
        perturb_actions_octo_curr = model.sample_actions(
            perturb_observation,
            task,
            unnormalization_statistics=model.dataset_statistics["bridge_dataset"][
                "action"
            ],
            rng=jax.random.PRNGKey(0),
        )

        perturb_actions_octo_curr = np.array(
            perturb_actions_octo_curr
        )  # [N, n_steps, 7]
        peturb_actions_octo_all.append(
            perturb_actions_octo_curr
        )  # Save for post-processing

        # Compute Delta
        delta = (
            nonperturb_actions_octo - perturb_actions_octo_curr
        )  # [num_samples, n_steps, 7]
        # Scale by w and square
        delta_sq = np.square(delta)  # [num_samples, n_steps, 7]
        delta_wsq = np.multiply(delta_sq, w)  # [num_samples, n_steps, 7]
        # Compute Magnitude
        delta_sum_wsq = np.sum(delta_wsq, axis=2)  # [num_samples, n_steps]
        delta_sqrt_sum_wsq = np.sqrt(delta_sum_wsq)  # [num_samples, n_steps]
        # Average over samples
        delta_avg_samples = np.mean(delta_sqrt_sum_wsq, axis=0)  # [n_steps]
        delta_final = np.mean(delta_avg_samples)  # scalar
        # print(f"Delta for {class_names[i]}: {delta_final}")
        delta_data_curr = {
            "object": class_names_sensitivity[i],
            "delta": delta,
            "delta_sq": delta_sq,
            "delta_wsq": delta_wsq,
            "delta_sum_wsq": delta_sum_wsq,
            "delta_sqrt_sum_wsq": delta_sqrt_sum_wsq,
            "delta_avg_samples": delta_avg_samples,
            "delta_final": delta_final,
        }
        delta_data[i] = delta_data_curr

        # Check Sensitivity
        if delta_final >= thresh:
            sensitivity[0, i] = True
        else:
            sensitivity[0, i] = False

    return sensitivity, delta_data


def background_sensitivities(
    original_img,
    class_names_sensitivity,
    detections_sensitivity,
    w,
    thresh,
    N,
    language_instruction,
    perturb_std,
    save_gs2_directory,
):
	"""
	Determines sensitivity of VLA to background regions
	"""
    num_objects = len(class_names_sensitivity)
    sensitivity = np.full((1, num_objects), True, dtype=bool)

    nonperturb_actions_octo = []  # holds all N samples
    peturb_actions_octo_all = (
        []
    )  

    delta_data = {}  # holds all deltas for k num_objects

    start_sensitivity = time.time()
    nonperturb_img = original_img.copy()
    nonperturb_img = warm_filter(nonperturb_img)
    nonperturb_img = cv2.resize(nonperturb_img, (256, 256))
    nonperturb_img = nonperturb_img[np.newaxis, np.newaxis, ...]
    nonperturb_img_curr = nonperturb_img.copy()

    texts = []
    timestep_pad_mask = []
    for n in range(N):
        if n == 0:
            nonperturb_img_curr = nonperturb_img_curr
            texts.append(language_instruction)
            timestep_pad_mask.append([True])
        else:
            nonperturb_img_curr = np.vstack([nonperturb_img_curr, nonperturb_img])
            texts.append(language_instruction)
            timestep_pad_mask.append([True])

    nonperturb_observation = {
        "image_primary": nonperturb_img_curr,
        "timestep_pad_mask": np.array(timestep_pad_mask),
    }

    task = model.create_tasks(texts=texts)

    nonperturb_actions_octo = model.sample_actions(
        nonperturb_observation,
        task,
        unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"],
        rng=jax.random.PRNGKey(0),
    )
    # Turn into array
    nonperturb_actions_octo = np.array(nonperturb_actions_octo)  # [N, n_steps, 7]

    # Determine which objects octo is sensitve to
    perturb_images = []
    perturb_actions_all_items_curr = []
    for i in range(num_objects):
        mask = get_mask(
            class_names_sensitivity[i],
            class_names_sensitivity,
            detections_sensitivity,
        )
        dilate_size = 0
        mask_dilate = dilate_mask(mask, dilate_size)

        texts = []
        timestep_pad_mask = []
        for n in range(N):
            if n == 0:

                # Perturb
                perturb_img = perturb_gaussian_noise(
                    original_img, mask_dilate, std=perturb_std
                )
                # perturb_img = perturb_gaussian_noise(original_img, mask_dilate)
                perturb_img_curr = warm_filter(perturb_img)

                # Save Image for Analysis
                perturb_images.append(perturb_img_curr)

                perturb_img_curr = cv2.resize(perturb_img_curr, (256, 256))
                perturb_img_curr = perturb_img_curr[np.newaxis, np.newaxis, ...]
                texts.append(language_instruction)
                timestep_pad_mask.append([True])
            else:
                random_kernel_size = np.random.randint(15, 30)

                # Perturb
                perturb_img = perturb_gaussian_noise(
                    original_img, mask_dilate, std=perturb_std
                )
                # perturb_img = perturb_gaussian_noise(original_img, mask_dilate)
                perturb_img = warm_filter(perturb_img)

                # Save Image for Analysis
                perturb_images.append(perturb_img)
                
                perturb_img = cv2.resize(perturb_img, (256, 256))
                perturb_img = perturb_img[np.newaxis, np.newaxis, ...]

                perturb_img_curr = np.vstack([perturb_img_curr, perturb_img])
                texts.append(language_instruction)
                timestep_pad_mask.append([True])

        perturb_observation = {
            "image_primary": perturb_img_curr,
            "timestep_pad_mask": np.array(timestep_pad_mask),
        }
        task = model.create_tasks(texts=texts)

        perturb_actions_octo_curr = []  # holds all N samples for current object
        perturb_actions_octo_curr = model.sample_actions(
            perturb_observation,
            task,
            unnormalization_statistics=model.dataset_statistics["bridge_dataset"][
                "action"
            ],
            rng=jax.random.PRNGKey(0),
        )

        perturb_actions_octo_curr = np.array(
            perturb_actions_octo_curr
        )  # [N, n_steps, 7]
        peturb_actions_octo_all.append(
            perturb_actions_octo_curr
        )  # Save for post-processing

        # Compute Delta
        delta = (
            nonperturb_actions_octo - perturb_actions_octo_curr
        )  # [num_samples, n_steps, 7]
        # Scale by w and square
        delta_sq = np.square(delta)  # [num_samples, n_steps, 7]
        delta_wsq = np.multiply(delta_sq, w)  # [num_samples, n_steps, 7]
        # Compute Magnitude
        delta_sum_wsq = np.sum(delta_wsq, axis=2)  # [num_samples, n_steps]
        delta_sqrt_sum_wsq = np.sqrt(delta_sum_wsq)  # [num_samples, n_steps]
        # Average over samples
        delta_avg_samples = np.mean(delta_sqrt_sum_wsq, axis=0)  # [n_steps]
        delta_final = np.mean(delta_avg_samples)  # scalar
        # print(f"Delta for {class_names[i]}: {delta_final}")
        delta_data_curr = {
            "object": class_names_sensitivity[i],
            "delta": delta,
            "delta_sq": delta_sq,
            "delta_wsq": delta_wsq,
            "delta_sum_wsq": delta_sum_wsq,
            "delta_sqrt_sum_wsq": delta_sqrt_sum_wsq,
            "delta_avg_samples": delta_avg_samples,
            "delta_final": delta_final,
        }
        delta_data[i] = delta_data_curr

        # Check Sensitivity
        if delta_final >= thresh:
            sensitivity[0, i] = True
        else:
            sensitivity[0, i] = False

        print(
            f"Delta for {class_names_sensitivity[i]}: {delta_final}. Inpaint: {sensitivity[0, i]}"
        )
        print("")
    return sensitivity, delta_data


def inpaint_objects(
    class_names_sensitivity, detections_sensitivity, sensitivity, img, dilate_size_vla
):
    sensitive_counter = 0
    for i in range(len(class_names_sensitivity)):
        curr_name = class_names_sensitivity[i]
        curr_sensitive_index = class_names_sensitivity.index(curr_name)
        sensitive = sensitivity[0, curr_sensitive_index]
        mask = get_mask(curr_name, class_names_sensitivity, detections_sensitivity)

        # Outpaint - make sure img is np array
        dilate_size = dilate_size_vla
        mask_dilate = dilate_mask(mask, dilate_size)

        # Call Inpaint Anything if Above Threshold
        if sensitive:
            if sensitive_counter == 0:
                mask_curr = mask_dilate
                sensitive_counter += 1
            else:
                mask_curr = np.logical_or(mask_curr, mask_dilate).astype(np.uint8)

    # print(f"Calling Inpaint Anything: {curr_name}")
    print("")
    if sensitive_counter > 0:
        img = outpaint_anything(img, mask_curr)
    else:
        img = img

    return img


def inpaint_backgrounds(
    class_names, detections, perturb_std, img, w, N, n_steps, thresh, curr_sensitivity
):
    inpainted_colors = False
    # curr_sensitivity = np.full((1, len(class_names)), False, dtype=bool)
    class_names_sensitivity_index = []
    color_index = {}
    inpaint_colors = {}
    color_sensitivity = {}
    inpaint_delta_data = {}

    for i in range(len(class_names)):
        # curr_name = matching[i]
        curr_name = class_names[i]
        sensitive = curr_sensitivity[0, i]
        curr_classname_index = i

        mask = get_mask(curr_name, class_names, detections)
        if save_gs2_annotations:
            mask_fname = save_gs2_directory + "_mask_" + class_names[i] + ".jpg"
            cv2.imwrite(mask_fname, mask.astype(np.uint8) * 255)
        # Outpaint - make sure img is np array
        dilate_size = 0
        mask_dilate = dilate_mask(mask, dilate_size)

        # Call Inpaint Anything if Above Threshold
        if sensitive:
            print(f"Calling Inpaint Anything: {curr_name}")
            still_sensitive = True
            if not inpainted_colors:
                while still_sensitive:
                    # Inpaint with random color
                    inpaint_img_original, curr_rgb = inpaint_random_color(
                        img, mask_dilate
                    )
                    # Check perturbation
                    inpaint_img = warm_filter(inpaint_img_original)
                    inpaint_img = cv2.resize(inpaint_img, (256, 256))
                    inpaint_img = inpaint_img[np.newaxis, np.newaxis, ...]
                    inpaint_observation = {
                        "image_primary": inpaint_img,
                        "timestep_pad_mask": np.array([[True]]),
                    }

                    inpaint_actions_octo_curr = (
                        []
                    )  # holds all N samples for current object
                    for j in range(N):
                        inpaint_actions_octo_j = model.sample_actions(
                            inpaint_observation,
                            task,
                            unnormalization_statistics=model.dataset_statistics[
                                "bridge_dataset"
                            ]["action"],
                            rng=jax.random.PRNGKey(j),
                        )
                        inpaint_actions_octo_j = np.array(inpaint_actions_octo_j[0])
                        inpaint_actions_octo_j_nstep = inpaint_actions_octo_j[
                            :n_steps, :
                        ]
                        inpaint_actions_octo_curr.append(inpaint_actions_octo_j_nstep)

                    inpaint_actions_octo_curr = np.array(
                        inpaint_actions_octo_curr
                    )  # [N, n_steps, 7]
                  

                    # Now, add noise, just like before
                    # Inpaint with random color
                    perturb_inpaint_img_original = perturb_gaussian_noise(
                        inpaint_img_original, mask_dilate, std=perturb_std
                    )
                    # Check perturbation
                    perturb_inpaint_img = warm_filter(perturb_inpaint_img_original)
                    perturb_inpaint_img = cv2.resize(perturb_inpaint_img, (256, 256))
                    perturb_inpaint_img = perturb_inpaint_img[
                        np.newaxis, np.newaxis, ...
                    ]
                    perturb_inpaint_observation = {
                        "image_primary": perturb_inpaint_img,
                        "timestep_pad_mask": np.array([[True]]),
                    }

                    perturb_inpaint_actions_octo_curr = (
                        []
                    )  # holds all N samples for current object
                    for j in range(N):
                        perturb_inpaint_actions_octo_j = model.sample_actions(
                            perturb_inpaint_observation,
                            task,
                            unnormalization_statistics=model.dataset_statistics[
                                "bridge_dataset"
                            ]["action"],
                            rng=jax.random.PRNGKey(j),
                        )
                        perturb_inpaint_actions_octo_j = np.array(
                            perturb_inpaint_actions_octo_j[0]
                        )
                        perturb_inpaint_actions_octo_j_nstep = (
                            perturb_inpaint_actions_octo_j[:n_steps, :]
                        )
                        perturb_inpaint_actions_octo_curr.append(
                            perturb_inpaint_actions_octo_j_nstep
                        )

                    perturb_inpaint_actions_octo_curr = np.array(
                        perturb_inpaint_actions_octo_curr
                    )  # [N, n_steps, 7]
                   

                    # Compute Delta
                    delta_inpaint = (
                        perturb_inpaint_actions_octo_curr - inpaint_actions_octo_curr
                    )  # [num_samples, n_steps, 7]
                    # Scale by w and square
                    delta_sq_inpaint = np.square(
                        delta_inpaint
                    )  # [num_samples, n_steps, 7]
                    delta_wsq_inpaint = np.multiply(
                        delta_sq_inpaint, w
                    )  # [num_samples, n_steps, 7]
                    # Compute Magnitude
                    delta_sum_wsq_inpaint = np.sum(
                        delta_wsq_inpaint, axis=2
                    )  # [num_samples, n_steps]
                    delta_sqrt_sum_wsq_inpaint = np.sqrt(
                        delta_sum_wsq_inpaint
                    )  # [num_samples, n_steps]
                    # Average over samples
                    delta_avg_samples_inpaint = np.mean(
                        delta_sqrt_sum_wsq_inpaint, axis=0
                    )  # [n_steps]
                    delta_final_inpaint = np.mean(delta_avg_samples_inpaint)  # scalar
                    print(f"Delta for {class_names[i]}: {delta_final_inpaint}")
                    inpaint_delta_data_curr = {
                        "object": curr_name,
                        "delta_inpaint": delta_inpaint,
                        "delta_sq_inpaint": delta_sq_inpaint,
                        "delta_wsq_inpaint": delta_wsq_inpaint,
                        "delta_sum_wsq_inpaint": delta_sum_wsq_inpaint,
                        "delta_sqrt_sum_wsq_inpaint": delta_sqrt_sum_wsq_inpaint,
                        "delta_avg_samples_inpaint": delta_avg_samples_inpaint,
                        "delta_final_inpaint": delta_final_inpaint,
                    }
                    inpaint_delta_data[curr_classname_index] = inpaint_delta_data_curr

                    # Check Sensitivity
                    if delta_final_inpaint >= thresh:
                        # if delta_final_inpaint - thresh >= epsilon:
                        still_sensitive = True

                    else:
                        still_sensitive = False

                # Set to inpaint image and continue
                img = inpaint_img_original
   
                inpaint_colors[curr_classname_index] = curr_rgb

                
                ###############################################################
            else:
                color = inpaint_colors[curr_classname_index]
                img = inpaint_specific_color(img, mask_dilate, color)

    inpainted_colors = True
    return img


# Environment Variables

device = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.insert(1, "Path to /Inpaint_Anything")
sys.path.insert(1, "Path to /Inpaint_Anything/segment_anything")
sys.path.insert(1, "Path to /Grounded-SAM-2/")
sys.path.insert(1, "Path to /Grounded-SAM-2/utils_groundedSAM2") # Change to avoid double utils
sys.path.insert(1, "Path to /Grounded-SAM-2/sam2")


# grounded SAM2
sam2_checkpoint = "Path to /Grounded-SAM-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino from huggingface
model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
    device
)

# lama
lama_model = "Path to /Inpaint_Anything/pretrained_models/big-lama"
lama_config = (
    "Path to /Inpaint_Anything/lama/configs/prediction/default.yaml"
)

# GPU
# Set to prevent overallocation of memory on GPU
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

if __name__ == "__main__":
    # Run
    # Create WidowX Instance
    bot = "CREATE ROBOT INSTANCE"

    # Camera Setup
	pipeline = True # initialize your specific camera

    # Environment Vars
    language_instruction = "Language Instruction"
    n_steps = 4  # action chunk
    model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")
    task = model.create_tasks(texts=[language_instruction])

    img_history = []
    action_history = []
    step = 0
    steps_max = 100  # arbitrary cut off point

    instruction_underscore = language_instruction.replace(" ", "_")
    thresh = 0.002 # sensitivity threshold for toy kitchen environment - change based upon setup
    w = np.array([1, 1, 1, 0, 0, 0, 0])  # weights for perturbation - we penalize only translational
    N = 5  # number of calls to average over for perturbation
    random_int = 0
    dilate_size_vla = 10
    trial_num = 1
    perturb_std = 0.075
    method = "method_name"
	
	# Create folder to save results
    experiment_directory = "/Experiments/"
    environment_directory = experiment_directory + instruction_underscore + "/"
    method_directory = environment_directory + method + "/"
    trial_directory = method_directory + "trial_" + str(trial_num) + "/"

    new_trial = True
    if not os.path.exists(trial_directory):
        os.makedirs(trial_directory)
    else:
        print("Trial already exists!")
        new_trial = False  # safety feature to prevent overwriting
        exit()

	# Move robot into initial position

    init_img = take_picture(pipeline)
    init_img_fname = trial_directory + "init_img.jpg"
    plt.imsave(init_img_fname, init_img)

    img_history_fname = trial_directory + "images.pkl"
    action_history_fname = trial_directory + "actions.pkl"

    # Call VLM
    vlm_output = gpt4o(init_img_fname, language_instruction)

    while new_trial and step < steps_max:
        print(f"Running Step: {step}")

        # Take Picture and Save
        img = take_picture(pipeline)
        img_fname = trial_directory + "Original" + str(step) + ".jpg"
        plt.imsave(img_fname, img)

        not_relevant_objects_list, not_relevant_backgrounds_list = vlm_refine_output(
            vlm_output
        )

        save_gs2_annotations = True
        save_gs2_directory = (
            trial_directory + "Foundation_Model_Outputs_Sensitivity/" + str(step) + "/"
        )
        if not os.path.exists(save_gs2_directory):
            os.makedirs(save_gs2_directory)

        gs2_object_input = grounded_sam2_text(not_relevant_objects_list)
        gs2_background_input = grounded_sam2_text(not_relevant_backgrounds_list)

        detections_objects, class_names_objects = grounded_sam2(
            img_fname, gs2_object_input, save_gs2_annotations, save_gs2_directory
        )


        # Calculate object sensitivities
        sensitivity_object, delta_data_object = object_sensitivities(
            img,
            class_names_objects,
            detections_objects,
            w,
            thresh,
            N,
            language_instruction,
        )

        # Inpaint Objects
        img = inpaint_objects(
            class_names_objects,
            detections_objects,
            sensitivity_object,
            img,
            dilate_size_vla,
        )

        img_fname = trial_directory + "ObjInpainted" + str(step) + ".jpg"
        plt.imsave(img_fname, img)

        detections_background, class_names_background = grounded_sam2(
            img_fname, gs2_background_input, save_gs2_annotations, save_gs2_directory
        )

        # Calculate background sensitivities
        sensitivity_background, delta_data_background = background_sensitivities(
            img,
            class_names_background,
            detections_background,
            w,
            thresh,
            N,
            language_instruction,
            perturb_std,
            save_gs2_directory,
        )

        # Inpaint Backgrounds
        img = inpaint_backgrounds(
            class_names_background,
            detections_background,
            perturb_std,
            img,
            w,
            N,
            n_steps,
            thresh,
            sensitivity_background,
        )

        img_fname = trial_directory + "Step" + str(step) + "_NoWarm.jpg"
        plt.imsave(img_fname, img)

        img = warm_filter(img)
       
        # Resize for Octo/OpenVLA
        img = cv2.resize(img, (256, 256))
        img = img[np.newaxis, np.newaxis, ...]

        # Create octo inputs
        observation = {
            "image_primary": img,
            "timestep_pad_mask": np.array([[True]]),
        }
        task = model.create_tasks(texts=[language_instruction])

        print("")
        random_int = step // n_steps # we found that changing the random key at every n_steps improved performance of base Octo policy
   
        # Run Octo
        actions_octo = model.sample_actions(
            observation,
            task,
            unnormalization_statistics=model.dataset_statistics["bridge_dataset"][
                "action"
            ],
            rng=jax.random.PRNGKey(random_int),
        )

        # Post Process Octo Actions
        actions_octo = np.array(actions_octo[0])
        actions_nstep = actions_octo[:n_steps, :]  # [n_steps,7]

        for i in range(n_steps):
            # Get curent actions
            action_curr = actions_nstep[i, :]
            actions = action_curr.tolist()
            action_history.append(actions)
            with open(action_history_fname, "wb") as f:
                pickle.dump(action_history, f)

            # Execute the action
            dx = actions[0]
            dy = actions[1]  
            dz = actions[2]
            dyaw = actions[3] 
            dpitch = actions[4]
            droll = actions[5]
            dgrasp = actions[6]


            bot.move_relative_ee(
                x=dx, y=dy, z=dz, roll=droll, pitch=dpitch, yaw=dyaw
            )

            epsilon = 0.7
            if dgrasp >= epsilon:
                bot.gripper.open()
            else:
                bot.gripper.close()
            step += 1

            # Take Picture (eval)
            if not (i == n_steps - 1):
                img = take_picture(pipeline)
                img_fname = trial_directory + "Original" + str(step) + ".jpg"
                plt.imsave(img_fname, img)

               