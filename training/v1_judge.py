"""train_v1 VLM judge (train_v1.md §4.2–§4.5). Qwen3-VL-4B-FP8 via vLLM, run once per collection.

Input: JSONL manifest, one row per chunk:
  {"id", "task", "real_start", "real_end", "dream": npz with frames[0]=dream frame 0, frames[1]=dream frame t_k}
  real_* are {"npz": path, "key": "image"} or {"npy": path}.
Output: JSONL with P(yes) for six questions per chunk:
  held_real, held_dream, placed_real, placed_dream (single end frame),
  placedpair_real, placedpair_dream (start + end frames).
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

MODEL = "~/hf_cache/models--Qwen--Qwen3-VL-4B-Instruct-FP8/snapshots/fefbb44cbcce8d1bb7e20b920b94f77432b3446d"

# task -> (target phrase, receptacle phrase, relation). Relation follows the official success term
# (object_in_container -> "inside", object_on_top -> "on").
NAMES = {
    "BananaInBowlTask": ("yellow banana", "red bowl", "inside"),
    "BananaOnPlateTask": ("yellow banana", "white plate", "on"),
    "CannedFoodInBinTask": ("small tuna can", "grey bin", "inside"),
    "CoffeePotInBinTask": ("black coffee pot", "grey bin", "inside"),
    "BowlInBinTask": ("bowl", "grey bin", "inside"),
    "SauceBottlesCrateTask": ("red BBQ sauce bottle", "crate", "inside"),
    "YogurtInBowlTask": ("small red yogurt cup", "red bowl", "inside"),
    "ButterAboveRaisinTask": ("butter box", "raisin box", "on"),
    "LargerObjectRaisinBoxInBinTask": ("raisin box", "grey bin", "inside"),
    "SmallerObjectButterInBinTask": ("butter box", "grey bin", "inside"),
    "CookingPickPastaToolTask": ("pink pasta tool", "other utensil container (not the one it started in)", "inside"),
    "FoodPacking1BoxesTask": ("boxed food item", "bin", "inside"),
    "FoodPacking1CansTask": ("canned food item", "bin", "inside"),
    "FruitsOnionTask": ("red onion", "wooden bowl", "inside"),
    "FruitsOnionToPlateTask": ("red onion", "plate", "on"),
    "WoodSpatulaToBowlTask": ("wooden spatula", "wooden bowl", "inside"),
    "PinkSpoonInPotTask": ("pink spaghetti spoon", "pot", "inside"),
    "MustardAboveRaisinTask": ("mustard bottle", "raisin box", "on"),
    "RubiksCubeTask": ("Rubik's cube", "bowl", "inside"),
    "PutBowlOnShelfTopTask": ("white serving bowl", "shelf", "on"),
    "ClampInRightBinTask": ("spring clamp", "right bin", "inside"),
    "MustardInLeftBinTask": ("mustard bottle", "left bin", "inside"),
    "MustardInRightBinTask": ("mustard bottle", "right bin", "inside"),
    "MarkerInMugTask": ("marker", "mug", "inside"),
    "MouseOnKeyboardTask": ("computer mouse", "keyboard", "on"),
    "ThrowAwayAppleTask": ("red apple", "grey bin", "inside"),
}

VIEWS = "In each image the top view is from the wrist camera, and the two lower views are side cameras. "
RULE = "Judge the visible state only. Do not infer success from the intended motion. Answer with only Yes or No.\n"
PAIR = ("The first image is from the start of a short robot motion and the second image is from its end. "
        "Answer about the second image; use the first only to track the object. ")


def questions(task):
    target, receptacle, relation = NAMES[task]
    return {
        "held": f"Is the robot currently holding the {target} in its gripper?",
        "placed": f"Is the {target} resting {relation} the {receptacle} and released from the gripper?",
    }


def load(ref):
    if "npz" in ref:
        with np.load(ref["npz"]) as data:
            array = data[ref["key"]]
            if "index" in ref:
                array = array[ref["index"]]
    else:
        array = np.load(ref["npy"])
    return Image.fromarray(np.asarray(array, dtype=np.uint8)).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--samples-dir", type=Path)
    parser.add_argument("--sample-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.15)
    args = parser.parse_args()
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    yes, no = (processor.tokenizer.encode(word, add_special_tokens=False)[0] for word in ("Yes", "No"))
    model = LLM(model=MODEL, max_model_len=4096, enable_prefix_caching=True, enforce_eager=True,
                gpu_memory_utilization=args.gpu_memory_utilization, limit_mm_per_prompt={"image": 2, "video": 0})
    params = SamplingParams(temperature=0, max_tokens=1, logprob_token_ids=[yes, no])
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    rng = random.Random(args.seed)
    if args.samples_dir:
        args.samples_dir.mkdir(parents=True, exist_ok=True)

    def prompt(images, text):
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
        return {"prompt": processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                         add_generation_prompt=True),
                "multi_modal_data": {"image": images if len(images) > 1 else images[0]}}

    with args.out.open("w") as output:
        for offset in range(0, len(rows), args.batch):
            started = time.monotonic()
            batch = rows[offset:offset + args.batch]
            requests, slots = [], []
            for row in batch:
                bank = questions(row["task"])
                real_start, real_end = load(row["real_start"]), load(row["real_end"])
                dream_start = load({"npz": row["dream"], "key": "frames", "index": 0})
                dream_end = load({"npz": row["dream"], "key": "frames", "index": 1})
                for side, start, end in (("real", real_start, real_end), ("dream", dream_start, dream_end)):
                    requests.append(prompt([end], VIEWS + RULE + bank["held"]))
                    slots.append((row["id"], f"held_{side}"))
                    requests.append(prompt([end], VIEWS + RULE + bank["placed"]))
                    slots.append((row["id"], f"placed_{side}"))
                    requests.append(prompt([start, end], PAIR + VIEWS + RULE + bank["placed"]))
                    slots.append((row["id"], f"placedpair_{side}"))
                if args.samples_dir and rng.random() < args.sample_fraction:
                    tiles = [image.resize((320, 270)) for image in (real_start, real_end, dream_start, dream_end)]
                    sheet = Image.new("RGB", (640, 540))
                    for index, tile in enumerate(tiles):
                        sheet.paste(tile, ((index % 2) * 320, (index // 2) * 270))
                    sheet.save(args.samples_dir / f"{row['id']}.jpg", quality=85)
            results = model.generate(requests, params, use_tqdm=False)
            answers = {}
            for (chunk, name), result in zip(slots, results, strict=True):
                entry = result.outputs[0].logprobs[0]
                score = entry[yes].logprob - entry[no].logprob
                answers.setdefault(chunk, {})[name] = 1 / (1 + math.exp(-score))
                answers[chunk][f"{name}_mass"] = math.exp(entry[yes].logprob) + math.exp(entry[no].logprob)
            for row in batch:
                print(json.dumps({"id": row["id"], **answers[row["id"]]}), file=output, flush=True)
            print(f"{offset + len(batch)}/{len(rows)} chunks in {time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
