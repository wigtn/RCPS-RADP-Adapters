# RCPS/RADP parser-training adapters

[![EMNLP 2026](https://img.shields.io/badge/EMNLP_2026-Industry_Track_·_Accepted-4F46E5?style=flat-square)](https://github.com/wigtn/WigtnOCR-RADP)
[![Release](https://img.shields.io/badge/Release-v1.0.0-181717?style=flat-square&logo=github)](https://github.com/wigtn/RCPS-RADP-Adapters/releases/tag/v1.0.0)

This release contains the nine final LoRA adapters evaluated in *Retrieval-Conditional
Parsing Score (RCPS): Choosing Document Parsers by Retrieval, Not by Appearance*,
accepted to the **EMNLP 2026 Industry Track** (Submission #384).
It includes the RADP-aux lambda sweep, RADP-DPO R1--R3,
RADP-Distill, and the SimPO control. These checkpoints support the paper's
secondary negative-result study; they are not a new production recommendation.
The paper, evaluation code, and deterministic release builder live in
[`wigtn/WigtnOCR-RADP`](https://github.com/wigtn/WigtnOCR-RADP).

The main evaluation base is [`Wigtn/Qwen3-VL-2B-WigtnOCR`](https://huggingface.co/Wigtn/Qwen3-VL-2B-WigtnOCR).
RADP-DPO, RADP-Distill, and SimPO were trained from that base. RADP-aux was
trained from [`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) and
then evaluated in the paper by applying its adapter to the Prod base. The
per-variant `adapter_config.json` records the training base, while `manifest.json`
records both training and evaluation bases.

## Where these adapters fit

| Step | Repository | What it holds |
|---|---|---|
| 1 | [`wigtn/wigtnOCR-v1`](https://github.com/wigtn/wigtnOCR-v1) | Distillation of Qwen3-VL-30B pseudo-labels into `Wigtn/Qwen3-VL-2B-WigtnOCR`, the base these adapters load onto (called *Prod* in the paper) |
| 2 | [`wigtn/WigtnOCR-RADP`](https://github.com/wigtn/WigtnOCR-RADP) | The RCPS paper, evaluation code, the 663-Q–A KoGovDoc-RAG probe, and result artifacts |
| 3 | This repository | The nine LoRA adapters from the paper's secondary parser-training study |

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
mkdir -p /path/to/RCPS-RADP-Adapters
tar -xzf RCPS-RADP-Adapters-v1.tar.gz -C /path/to/RCPS-RADP-Adapters
```

The release tarball is 358,728,348 bytes with SHA-256
`22b4a0d4f5560f4d7c31633a1c885bbe3568a2bef841bfd94bf2407c339f08c6`.

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
See the main repository's
[`CHECKPOINT_RELEASE.md`](https://github.com/wigtn/WigtnOCR-RADP/blob/main/docs/provenance/CHECKPOINT_RELEASE.md)
for the inclusion policy and CPU-only verification command.

## Citation

```bibtex
@inproceedings{son2026rcps,
  title     = {Retrieval-Conditional Parsing Score (RCPS): Choosing Document Parsers by Retrieval, Not by Appearance},
  author    = {Son, Sang-Woo and Kim, Hyeong-seob and Kim, Hyeonsang and Cho, Hyun-woo and Kim, Jinmo},
  booktitle = {Proceedings of EMNLP 2026 (Industry Track)},
  year      = {2026},
  note      = {Accepted}
}
```
