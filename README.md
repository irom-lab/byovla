# Bring Your Own VLA

[[Paper]](https://aasherh.github.io/data/Hancock_Visually_Robust_VLAs.pdf)   [[Website]](https://aasherh.github.io/byovla/)

[Asher J. Hancock<sup>1</sup>](https://aasherh.github.io/), [Allen Z. Ren<sup>1</sup>](https://allenzren.github.io/), [Anirudha Majumdar<sup>1</sup>](https://irom-lab.princeton.edu/majumdar/)

<sup>1</sup>Princeton University

<img src="https://github.com/AasherH/byovla/blob/main/img/anchor_figure.png" alt="drawing" width="100%"/>

> We introduce Bring Your Own VLA (BYOVLA): a run-time intervention scheme for vision-language-action (VLA) models that improves baseline performance in the presence of distractor objects and backgrounds without finetuning or access to the model's weights.

## Getting Started and Usage

For example, to utilize `BYOVLA` on [Octo-Base](https://github.com/octo-models/octo) with a WidowX robot in an environment from the [BridgeV2 Dataset](https://github.com/rail-berkeley/bridge_data_v2):

```python
sam2_checkpoint = "/Grounded-SAM-2/checkpoints/sam2_hiera_large.pt"
sam2model = build_sam2("sam2_hiera_l.yaml", sam2_checkpoint, device='cuda')
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino from huggingface
model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
    device
)

lama_model = "Path to /Inpaint_Anything/pretrained_models/big-lama"
lama_config = (
    "Path to /Inpaint_Anything/lama/configs/prediction/default.yaml"
)

# create widowx instance
bot = InterbotixManipulatorXS("wx250s", "arm", "gripper")

# load VLA
language_instruction = "place the carrot on plate"
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")
task = model.create_tasks(texts=[language_instruction])

# initial observation
img = take_picture()

# Step 1: call VLM
vlm_output = gpt4o(img, language_instruction)
not_relevant_objects_list = vlm_refine_output(vlm_output)

# call segmentation model on objects
gs2_object_input = grounded_sam2_text(not_relevant_objects_list)
detections_objects, class_names_objects = grounded_sam2(img, gs2_object_input)

# Step 2: compute vla sensitivities
sensitivities = object_sensitivities(img, class_names_objects, detections_objects,model, language_instruction)

# Step 3: transform image
img = inpaint_objects(class_names_objects,detections_objects,sensitivity_object,img)

# call vla on transformed image
observation = {
    "image_primary": img,
    "timestep_pad_mask": np.array([[True]])}
task = model.create_tasks(texts=[language_instruction])

actions = model.sample_actions(
            observation,
            task,
            unnormalization_statistics=model.dataset_statistics["bridge_dataset"][
                "action"],
            rng=jax.random.PRNGKey(0))

# execute action
bot.act(action, ...)
```
## Installation
Use the setup commands below to get started. The exact foundation models used are user-dependent, but we provide instructions for the ones utilized in this work. 

```python

# Clone and install byovla repository
git clone https://github.com/irom-lab/byovla.git
cd byovla

# clone Grounded-SAM2
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git

# clone Inpaint-Anything
git clone https://github.com/geekyutao/Inpaint-Anything.git

# clone VLA - we use Octo, for example
git clone https://github.com/octo-models/octo.git

# Create conda environment and install environment.yaml file
conda env create -f environment.yml

# activate environment
conda activate byovla
```

This code is meant to be run on the WidowX robot and will not work without first following the instructions [here](https://docs.trossenrobotics.com/interbotix_xsarms_docs/getting_started.html). 


Follow the instructions for downloading [Grounded-SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2). Then, follow the instructions for installing [Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything), paying special attention to `sam` and `lama` checkpoints. Change `./Grounded-SAM-2/utils/` to `./Grounded-SAM-2/utils_groundedSAM2/` to avoid import errors, as different packages both used the name `utils`. Change any other paths needed in your environment. 

If you run into any problems during the installation process, please file a GitHub Issue and we will strive to address it soon. 

## Notes
`BYOVLA` is a run-time observation intervention scheme and may be used at every time-step for which there is a new observation. In our experiments, we used the RealSense D435 camera, but you may need to install additional dependencies for your specific camera. 
The code provided is a skeleton and requires the user to change file paths, api keys, etc. 

## License
This repository is released under the MIT license. See [LICENSE](LICENSE).

## Acknowledgement
* [Octo, Octo Model Team](https://github.com/octo-models/octo): VLA model
* [OpenVLA, Kim et al.](https://github.com/openvla/openvla): VLA model
* [Grounded-SAM 2, Ren et al.](https://github.com/IDEA-Research/Grounded-SAM-2): code for grounded segmentation model
* [Inpaint Anything, Yu et al.](https://github.com/geekyutao/Inpaint-Anything?tab=readme-ov-file): code for inpainting model
