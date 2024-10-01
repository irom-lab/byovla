# Bring Your Own VLA

[[Paper]](https://aasherh.github.io/data/Hancock_Visually_Robust_VLAs.pdf)   [[Website]](https://aasherh.github.io/byovla.github.io/)

[Asher J. Hancock<sup>1</sup>](https://aasherh.github.io/), [Allen Z. Ren<sup>1</sup>](https://allenzren.github.io/), [Anirudha Majumdar<sup>1</sup>](https://irom-lab.princeton.edu/majumdar/)

<sup>1</sup>Princeton University

<img src="https://github.com/AasherH/byovla/blob/main/img/anchor_figure.png" alt="drawing" width="100%"/>

> We introduce Bring Your Own VLA (BYOVLA): a run-time intervention scheme for vision-language-action (VLA) models that improves baseline performance in the presence of distractor objects and backgrounds without finetuning or access to the model's weights.

## Getting Started

For example, to load `openvla-7b` for zero-shot instruction following in the
[BridgeData V2 environments](https://rail-berkeley.github.io/bridgedata/) with a WidowX robot:

```python
# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", 
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

# Grab image input & format prompt
image: Image.Image = get_from_camera(...)
prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

# Execute...
robot.act(action, ...)
```


