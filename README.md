# RCPS/RADP parser-training adapters

This release contains the nine final LoRA adapters evaluated in *Retrieval-Conditional
Parsing Score (RCPS): Choosing Document Parsers by Retrieval, Not by Appearance*.
It includes the RADP-aux lambda sweep, RADP-DPO R1--R3,
RADP-Distill, and the SimPO control. These checkpoints support the paper's
secondary negative-result study; they are not a new production recommendation.

The main evaluation base is [`Wigtn/Qwen3-VL-2B-WigtnOCR`](https://huggingface.co/Wigtn/Qwen3-VL-2B-WigtnOCR).
RADP-DPO, RADP-Distill, and SimPO were trained from that base. RADP-aux was
trained from [`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) and
then evaluated in the paper by applying its adapter to the Prod base. The
per-variant `adapter_config.json` records the training base, while `manifest.json`
records both training and evaluation bases.

## Variants

| Paper label | Subfolder |
|---|---|
| RADP-aux lambda=0.0 | `radp-aux/lambda-0.0` |
| RADP-aux lambda=0.1 | `radp-aux/lambda-0.1` |
| RADP-aux lambda=0.3 | `radp-aux/lambda-0.3` |
| RADP-aux lambda=0.5 | `radp-aux/lambda-0.5` |
| RADP-DPO-R1 | `radp-dpo/r1` |
| RADP-DPO-R2 | `radp-dpo/r2` |
| RADP-DPO-R3 | `radp-dpo/r3` |
| RADP-Distill | `radp-distill` |
| RADP-SimPO | `radp-simpo` |

## Loading an evaluated adapter

Download and extract the audited release first:

```bash
curl -L -o RCPS-RADP-Adapters-v1.tar.gz "https://github.com/wigtn/RCPS-RADP-Adapters/releases/download/v1.0.0/RCPS-RADP-Adapters-v1.tar.gz"
tar -xzf RCPS-RADP-Adapters-v1.tar.gz -C /path/to/RCPS-RADP-Adapters
```

```python
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration

base = Qwen3VLForConditionalGeneration.from_pretrained(
    "Wigtn/Qwen3-VL-2B-WigtnOCR", torch_dtype="auto", device_map="auto"
)
model = PeftModel.from_pretrained(
    base,
    "/path/to/RCPS-RADP-Adapters/radp-dpo/r2",
)
```

`manifest.json` gives original and release hashes, executed hyperparameters,
source-log hashes, and the source repository paths for every variant. Structured
`trainer_state.json` files are included where the Transformers trainer produced
them. Optimizer states, training images, and preference-pair text are excluded.
The repository release builder verifies the original weights before staging.
