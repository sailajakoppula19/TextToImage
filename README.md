# texttoimage
# Text-to-Image Internship Projects

> **Repo:** `texttoimage`
>
> **Author:** Sailaja Koppula
>
> **Internship Window:** 20 May 2025 – 20 July 2025 (Asia/Kolkata)
>
> **Scope:** A multi-task learning & engineering series covering text preprocessing, tokenization, conditional GAN image generation, an end‑to‑end text‑to‑image pipeline, and fine‑tuning a pre‑trained diffusion model — delivered with notebooks, scripts, saved models, metrics, and a demo UI.

---

## 🔖 Quick Links


  * [Task 1: Tokenization & Encoding](#task-1-tokenization--encoding)
  * [Task 2: Text Preprocessing for T2I](#task-2-text-preprocessing-for-t2i)
  * [Task 3: CGAN for Basic Shapes](#task-3-cgan-for-basic-shapes)
  * [Task 4: Full Text-to-Image Pipeline](#task-4-full-text-to-image-pipeline)
  * [Task 5: Fine-Tuning Stable Diffusion (Domain-Specific)](#task-5-fine-tuning-stable-diffusion-domain-specific)

---

## 🎯 Project Goals


1. **Text tokenization & numerical encoding** using pre‑trained language models (BERT/GPT tokenizers via Hugging Face).
2. **Text preprocessing for image generation prompts** — cleaning, normalization, truncation, vocab stats.
3. **Conditional GAN (CGAN)** to generate basic geometric shapes (circle, square, triangle) from class labels.
4. **End‑to‑end text‑to‑image mini‑pipeline** (tokenize → embed → condition generator → decoder / upsampler).
5. **Fine‑tuning a pre‑trained diffusion model** (Stable Diffusion/SD‑Lite) on a domain dataset (e.g., medical, artwork).

Goal: Clean, runnable notebooks + scripts + saved models + metrics + short report/documentation (this README!).

---

## 🛠 Environment Setup

### 1. Clone Repo

```bash
git clone https://github.com/sailajakoppula19/texttoimage.git
cd texttoimage
```

### 2. Create Environment (Conda Recommended)

```bash
conda create -n text2img python=3.10 -y
conda activate text2img
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> If using GPU + PyTorch

```bash
# Example; adjust CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```


---

## 📦 Data Preparation

Some datasets are big in size(especially domain images / Stable Diffusion fine‑tuning). Use the helper script or manual download.

### Option A: Scripted Download

```bash
bash scripts/download_data.sh --all
```

### Option B: Manual Google Drive Links

* **Shapes Dataset (Task 3):** \<ADD\_LINK>
* **Domain Dataset (Task 5 – e.g., Diabetic Retinopathy / Art style):** \<ADD\_LINK>
* **Prompt CSV / Text Data:** \<ADD\_LINK>

Put downloaded content under `data/` matching the folder names above.

### Data Sanity Check

```bash
python -m src.utils.io_utils --check-data
```

---

## ▶ Running the Tasks

Below are mini run guides. Detailed walkthroughs live inside the corresponding notebooks.

---

### Task 1: Tokenization & Encoding

**Goal:** Use a pre‑trained tokenizer (BERT/GPT/DistilBERT) to convert raw text prompts into token ID sequences, attention masks, and embeddings for downstream conditioning.

**Notebook:** `notebooks/01_tokenization_encoding.ipynb`

**Script Usage:**

```bash
python -m src.text.tokenize \
  --input data/raw/prompts.txt \
  --tokenizer bert-base-uncased \
  --max_length 64 \
  --output models/tokenizer/tokenized_prompts.pt
```

**Outputs:**

* `token_ids`, `attention_mask`, `token_type_ids` (if model supports)
* Vocab stats JSON (top tokens, unk rate)
* Saved tokenizer config & vocab

---

### Task 2: Text Preprocessing for T2I

**Goal:** Clean, normalize, and standardize prompts so that downstream models get consistent conditioning signals.

**Key Steps:** lowercase (optional), strip emoji or keep tags?, remove excessive punctuation, truncate or pad to max token length, optionally add class prefixes.

**Notebook:** `notebooks/02_text_preprocessing_t2i.ipynb`

**Script Example:**

```bash
python -m src.text.preprocess \
  --in_csv data/raw/prompts.csv \
  --out_csv data/processed/prompts_clean.csv \
  --min_words 2 --max_words 32 \
  --drop_dupes
```

**Artifacts:**

* Clean prompt CSV
* Token length histograms
* Before/after sample inspection

---

### Task 3: CGAN for Basic Shapes

**Goal:** Conditional GAN generate tiny 32×32 (or 64×64) images of **circle, square, triangle** given a class label embedding.

**Notebook:** `notebooks/03_cgan_shapes_train.ipynb`

**Train (CLI):**

```bash
python -m src.models_cgan.train_cgan \
  --data_dir data/shapes \
  --epochs 100 \
  --batch_size 128 \
  --latent_dim 100 \
  --num_classes 3 \
  --out_dir models/cgan
```

**Check Samples:**

```bash
python -m src.models_cgan.train_cgan --sample-only --checkpoint models/cgan/latest.pt
```

**Metrics:** Inception Score (toy), FID (optional), pixel overlap vs template masks (bonus).

---

### Task 4: Full Text-to-Image Pipeline

**Goal:** Build a lightweight research pipeline connecting text embeddings → conditional image generator → upsampler / decoder → output image.

**Supports:**

* Simple U-Net or VQGAN decoder
* Conditioning via concatenated embeddings or cross‑attn stub
* Scheduled sampling from latent noise

**Notebook:** `notebooks/04_full_text_to_image_pipeline.ipynb`

**Train:**

```bash
python -m src.pipeline.trainer \
  --prompts data/processed/prompts_clean.csv \
  --images data/processed/imgs \
  --epochs 20 \
  --img_size 128 \
  --save_dir models/t2i_pipeline
```

**Infer:**

```bash
python -m src.pipeline.infer \
  --prompt "a red circle on white background" \
  --checkpoint models/t2i_pipeline/best.pt \
  --out demo/outputs/red_circle.png
```

---

### Task 5: Fine-Tuning Stable Diffusion (Domain-Specific)

**Goal:** Adapt a pre‑trained text‑to‑image diffusion model (e.g., Stable Diffusion 1.5 / SDXL‑Lite / Latent Diffusion) to a narrow domain such as **Diabetic Retinopathy fundus images**, **medical lesions**, or **custom art style**.

**Methods Supported:**

* DreamBooth style few‑shot subject tuning
* LoRA / Low‑Rank Adapters (efficient fine‑tune)
* Textual Inversion (embedding token for new concept)

**Notebook:** `notebooks/05_sd_finetune_domain.ipynb`

**Example LoRA Fine‑Tune:**

```bash
accelerate launch -m src.sd_finetune.dreambooth_train \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-1-5 \
  --instance_data_dir data/domain/images \
  --class_prompt "retinal fundus photograph" \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --max_train_steps 2000 \
  --lora_rank 8 \
  --output_dir models/sd_finetune
```

**Inference After Fine‑Tune:**

```bash
python -m src.sd_finetune.infer_sd \
  --checkpoint models/sd_finetune \
  --prompt "mild diabetic retinopathy fundus image with few microaneurysms" \
  --num_images 4 \
  --out_dir demo/outputs/
```

---

## 🌐 Gradio / Web Demo

A lightweight Gradio UI allows interactive prompt → image generation across multiple trained backends.

**Launch:**

```bash
python demo/app.py --model t2i_pipeline --device cuda
```

**Switch Models:**

```bash
python demo/app.py --model sd_finetune --device cuda --height 512 --width 512
```

UI Features:

* Prompt text box
* Model selector dropdown (Token Demo / CGAN / Pipeline / SD Fine‑Tune)
* Num images slider
* Seed control for reproducibility
* Gallery output + download buttons

---

## ⚙ Training Tips & Hardware Notes

| Scenario | Suggested Batch    | Mixed Precision | Notes                                                    |
| -------- | ------------------ | --------------- | -------------------------------------------------------- |
| CPU only | 8–16               | fp32            | Slow; use tiny models.                                   |
| Colab T4 | 32 (CGAN) / 4 (SD) | fp16            | Use gradient accumulation.                               |
| 8GB GPU  | Up to 128×128 imgs | fp16/bf16       | Reduce model depth.                                      |
| 24GB GPU | 512px SD fine‑tune | fp16            | Enable xformers memory‑efficient attention if installed. |

*Tip:* Always set seeds for reproducibility: `python -m src.utils.seed_utils --seed 42` or call `seed_everything(42)` in notebooks.

---

## 📈 Evaluation & Metrics

Different tasks → different metrics. Here's what I logged:

| Task | Metric(s)                                             | File(s)                           |
| ---- | ----------------------------------------------------- | --------------------------------- |
| 1    | Token length dist, OOV rate                           | `reports/token_stats.json`        |
| 2    | Prompt retention %, drop count                        | `reports/preprocess_summary.csv`  |
| 3    | Inception Score (toy), FID (opt), class‑cond accuracy | `reports/cgan_metrics.csv`        |
| 4    | CLIP text‑image similarity, reconstruction loss       | `reports/pipeline_metrics.csv`    |
| 5    | CLIP Score vs base SD, domain expert rating           | `reports/sd_finetune_metrics.csv` |

Visualization notebooks generate plots into `reports/figs/`.

---

## ✅ Submission Artifacts Checklist

Use this when packaging your internship submission (20 May – 20 July 2025).

* [ ] **All 5 Task Notebooks** cleaned & runnable top‑to‑bottom.
* [ ] `requirements.txt` + optional `environment.yml`.
* [ ] Saved **tokenizer artifacts** (`models/tokenizer/`).
* [ ] **Preprocessed prompt CSV** & logs.
* [ ] **CGAN weights** + sample outputs.
* [ ] **T2I pipeline checkpoint(s)**.
* [ ] **Stable Diffusion fine‑tuned weights / LoRA** (or safetensors if large; link if >Git LFS cap).
* [ ] **Demo UI working** (Gradio) with at least sample inference.
* [ ] **Metrics CSVs** in `reports/`.
* [ ] **Internship report PDF** (if required by program).
* [ ] ZIP / Drive folder shared with mentors.

---

## 🧰 Common Issues & Fixes

**CUDA out of memory?** Reduce batch, image size, or enable gradient checkpointing.

**Tokenizer mismatch error?** Delete cached tokenizer dir & re‑download from Hugging Face.

**Images misaligned with prompts?** Ensure CSV row order matches image filenames; use `prompt_dataset.py --verify`.

**Colab disconnects?** Use `--save_every` flags so you can resume.

**xformers import error?** Reinstall with matching CUDA; else disable memory‑efficient attention.

---

## 🔮 Future Work

* Upgrade Task 4 pipeline to latent diffusion mini‑version.
* Add prompt weighting & negative prompts.
* Integrate safety checker for NSFW filtering.
* Add W\&B experiment logging.
* Expand medical domain (RetticScan fundus severity grading integration).

---

## 🙏 Acknowledgments

* Hugging Face Transformers & Diffusers ecosystems.
* PyTorch, Accelerate, TorchVision.
* Community datasets: (EyePACS / APTOS for Diabetic Retinopathy; confirm licensing before redistribution.)
* Nullclass Internship Program mentors & reviewers.

---

## 📫 Contact

If questions, open an **Issue** in this repo or ping me:

**Sailaja Koppula**
Email:sailajareddy762@gmail.com
LinkedIn/GitHub: `sailajakoppula19`

---


* **Tokenize** text → numbers.
* **Clean prompts** for training.
* **CGAN** shapes generate chestundi.
* **Mini text‑to‑image pipeline** build chesam.
* **Stable Diffusion fine‑tune** domain data meeda (ex: eye disease images).

Run notebooks order lo, data download chesi, `demo/app.py` run chey. Aithe ayipoindi! 😄

---

*Happy building & generating!* ✨🖼️
