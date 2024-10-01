# Bring Your Own VLA

[[Paper]](https://aasherh.github.io/data/Hancock_Visually_Robust_VLAs.pdf)   [[Website]](https://aasherh.github.io/byovla.github.io/)

[Asher J. Hancock<sup>1</sup>](https://aasherh.github.io/), [Allen Z. Ren<sup>1</sup>](https://allenzren.github.io/), [Anirudha Majumdar<sup>1</sup>](https://irom-lab.princeton.edu/majumdar/)

<sup>1</sup>Princeton University

<img src="https://github.com/AasherH/byovla/blob/main/img/anchor_figure.png" alt="drawing" width="100%"/>

> We introduce Bring Your Own VLA (BYOVLA): a run-time intervention scheme for vision-language-action (VLA) models that improves baseline performance in the presence of distractor objects and backgrounds without finetuning or access to the model's weights.

## Getting Started

For example, to utilize `BYOVLA` on Octo-Base
[Octo]([https://rail-berkeley.github.io/bridgedata/)](https://github.com/octo-models/octo) with a WidowX robot:

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


