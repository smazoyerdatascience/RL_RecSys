"""
generate_images.py
------------------
Generates one image per product via Stability AI (SDXL),
saves them locally in /images, and updates the CSV with a `filename` column.

Usage:
    python generate_images.py --csv products_named.csv --output images/ --start 0

Dependencies:
    pip install pandas requests tqdm

Environment variables:
    STABILITY_API_KEY=sk-...
"""

import os
import time
import base64
import argparse
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# Conf
API_URL = (
    "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
)

NEGATIVE_PROMPT = (
    "blurry, low quality, watermark, text overlay, logo, label, "
    "multiple products, person, human, hands, body parts, "
    "background clutter, dark shadows, overexposed, cartoon, illustration, "
    "painting, sketch, 3D render, deformed, disfigured"
)

# Prompts by category + subcategory + filled by product name
SUBCATEGORY_PROMPTS = {
    "electronics": {
        "smartphones": (
            "professional e-commerce product photo of a modern smartphone called {name}, "
            "sleek glass and metal design, screen on, front and back view, "
            "isolated on pure white background, studio lighting, sharp focus, 4K"
        ),
        "laptops": (
            "professional e-commerce product photo of a premium laptop called {name}, "
            "open lid showing keyboard and screen, slim aluminum body, "
            "isolated on pure white background, studio lighting, sharp focus, 4K"
        ),
        "tablets": (
            "professional e-commerce product photo of a modern tablet called {name}, "
            "thin profile, screen on displaying colorful interface, "
            "isolated on pure white background, studio lighting, 4K"
        ),
        "headphones": (
            "professional e-commerce product photo of premium headphones called {name}, "
            "over-ear or in-ear design, sleek modern look, "
            "isolated on pure white background, soft studio lighting, 4K"
        ),
        "cameras": (
            "professional e-commerce product photo of a camera called {name}, "
            "detailed lens and body, professional photography equipment look, "
            "isolated on pure white background, studio lighting, sharp details, 4K"
        ),
    },
    "fashion": {
        "clothing": (
            "professional fashion e-commerce product photo of {name}, "
            "flat lay on pure white background, fabric texture clearly visible, "
            "clean minimal style, studio lighting, high resolution, no model, no mannequin"
        ),
        "shoes": (
            "professional e-commerce product photo of {name} shoes, "
            "side profile view and top view, pair displayed together, "
            "isolated on pure white background, studio lighting, sharp details"
        ),
        "accessories": (
            "professional e-commerce product photo of a fashion accessory called {name}, "
            "detailed close-up showing material and craftsmanship, "
            "isolated on pure white background, soft studio lighting, high resolution"
        ),
        "jewelry": (
            "professional luxury e-commerce product photo of jewelry called {name}, "
            "gold or silver metallic finish, gemstone details if applicable, "
            "isolated on pure white background, macro studio lighting, sparkling, elegant"
        ),
    },
    "home_deco": {
        "furniture": (
            "professional interior design e-commerce product photo of {name} furniture, "
            "clean Scandinavian or modern style, natural wood or metal materials, "
            "isolated on pure white background, soft studio lighting, sharp details"
        ),
        "lighting": (
            "professional e-commerce product photo of a lamp or lighting fixture called {name}, "
            "light on, warm glow visible, elegant design, "
            "isolated on pure white background, studio lighting, high resolution"
        ),
        "decorative_items": (
            "professional interior decor e-commerce product photo of {name}, "
            "artisan or designer object, ceramic or natural materials, "
            "isolated on pure white background, soft natural studio lighting, elegant"
        ),
    },
    "sports": {
        "fitness_equipment": (
            "professional e-commerce product photo of fitness equipment called {name}, "
            "gym or home workout gear, clean industrial design, "
            "isolated on pure white background, dynamic studio lighting, high resolution"
        ),
        "outdoor_gear": (
            "professional e-commerce product photo of outdoor gear called {name}, "
            "hiking or trail equipment, technical materials, vibrant colors, "
            "isolated on pure white background, studio lighting, sharp details"
        ),
        "sportswear": (
            "professional e-commerce flat lay product photo of sportswear called {name}, "
            "technical fabric texture visible, athletic design, vibrant colors, "
            "on pure white background, studio lighting, no model"
        ),
    },
    "beauty": {
        "skincare": (
            "professional luxury beauty e-commerce product photo of skincare called {name}, "
            "elegant glass or tube packaging, minimalist label design, "
            "isolated on pure white background, soft glowing studio light, clean and elegant"
        ),
        "makeup": (
            "professional luxury makeup e-commerce product photo of {name}, "
            "sleek packaging, pigmented color swatches visible if applicable, "
            "isolated on pure white background, soft beauty studio lighting, elegant"
        ),
        "haircare": (
            "professional beauty e-commerce product photo of haircare product called {name}, "
            "elegant bottle or tube packaging, fresh and clean look, "
            "isolated on pure white background, soft studio lighting, high resolution"
        ),
    },
    "books": {
        "fiction": (
            "professional bookstore e-commerce product photo of a fiction novel called {name}, "
            "hardcover or paperback, artistic cover design with title clearly visible, "
            "isolated on pure white background, studio lighting, sharp and clear"
        ),
        "non-fiction": (
            "professional bookstore e-commerce product photo of a non-fiction book called {name}, "
            "clean professional cover design with title visible, "
            "isolated on pure white background, studio lighting, sharp and clear"
        ),
        "children_books": (
            "professional e-commerce product photo of a children's book called {name}, "
            "colorful illustrated cover, playful and fun design, "
            "isolated on pure white background, bright studio lighting, vibrant colors"
        ),
    },
    "food": {
        "snacks": (
            "professional gourmet food e-commerce product photo of {name}, "
            "artisan packaging or natural presentation, appetizing close-up, "
            "isolated on pure white background, natural studio lighting, high resolution"
        ),
        "beverages": (
            "professional gourmet beverage e-commerce product photo of {name}, "
            "elegant bottle, tin, or packaging, premium and artisan look, "
            "isolated on pure white background, studio lighting, appetizing"
        ),
        "gourmet_foods": (
            "professional gourmet food e-commerce product photo of {name}, "
            "premium jar, bottle or artisan packaging, French delicatessen style, "
            "isolated on pure white background, natural studio lighting, elegant and appetizing"
        ),
    },
    "toys_games": {
        "board_games": (
            "professional e-commerce product photo of a board game called {name}, "
            "box and components displayed, colorful and fun design, "
            "isolated on pure white background, bright studio lighting, child-friendly"
        ),
        "action_figures": (
            "professional e-commerce product photo of a toy called {name}, "
            "detailed figurine or vehicle, vibrant colors, fun design, "
            "isolated on pure white background, bright studio lighting"
        ),
        "puzzles": (
            "professional e-commerce product photo of a puzzle or construction toy called {name}, "
            "colorful pieces displayed attractively, educational and fun look, "
            "isolated on pure white background, bright studio lighting"
        ),
    },
}

# Fallback for unknown category/subcategory
DEFAULT_PROMPT = (
    "professional e-commerce product photo of {name}, "
    "isolated on pure white background, studio lighting, sharp focus, high resolution, 4K"
)

# Helpers


def build_prompt(name: str, category: str, subcategory: str) -> str:
    template = SUBCATEGORY_PROMPTS.get(category, {}).get(subcategory, DEFAULT_PROMPT)
    return template.format(name=name)


def generate_image(
    name: str, category: str, subcategory: str, api_key: str
) -> bytes | None:
    """Calls Stability AI and returns the PNG image bytes."""
    prompt = build_prompt(name, category, subcategory)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    body = {
        "text_prompts": [
            {"text": prompt, "weight": 1.0},
            {"text": NEGATIVE_PROMPT, "weight": -1.0},
        ],
        "cfg_scale": 7,
        "height": 1024,
        "width": 1024,
        "steps": 30,
        "samples": 1,
    }

    try:
        response = requests.post(API_URL, headers=headers, json=body, timeout=60)
        response.raise_for_status()
        data = response.json()
        image_b64 = data["artifacts"][0]["base64"]
        return base64.b64decode(image_b64)

    except requests.exceptions.HTTPError as e:
        print(f"\nHTTP error {response.status_code} for '{name}': {e}")
        if response.status_code == 429:
            print("   Rate limit reached, pausing 10s...")
            time.sleep(10)
        return None

    except Exception as e:
        print(f"\nUnexpected error for '{name}': {e}")
        return None


# Main


def main():
    parser = argparse.ArgumentParser(
        description="Generates product images via Stability AI"
    )
    parser.add_argument(
        "--csv", default="data/products_named.csv", help="Path to the products CSV"
    )
    parser.add_argument(
        "--output", default="data/images", help="Output directory for the images"
    )
    parser.add_argument("--start", type=int, default=0, help="Start index (to resume)")
    parser.add_argument(
        "--limit", type=int, default=None, help="Max number of products to process"
    )
    args = parser.parse_args()

    # API key
    api_key = os.environ.get("STABILITY_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "STABILITY_API_KEY is not set.\n"
            "Export it with: export STABILITY_API_KEY=sk-..."
        )

    # Load the CSV
    print(f"Loading {args.csv}...")
    df = pd.read_csv(args.csv)
    print(f"{len(df)} products loaded")

    # Create the filename column if it doesn't exist, and force object dtype
    # so it can hold strings even when every value is still NaN (fresh column,
    # or resumed run where no image succeeded yet)
    if "filename" not in df.columns:
        df["filename"] = pd.Series([None] * len(df), dtype="object")
    else:
        df["filename"] = df["filename"].astype("object")

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Images saved to: {output_dir.resolve()}\n")

    # Preview a prompt for verification before running
    first = df.iloc[args.start]
    print("Example generated prompt:")
    print(f"   Product  : {first['name']}")
    print(f"   Category : {first['category']} / {first['subcategory']}")
    print(
        f"   Prompt   : {build_prompt(first['name'], first['category'], first['subcategory'])}"
    )
    print()

    # Select the range to process
    subset = df.iloc[args.start :]
    if args.limit:
        subset = subset.head(args.limit)

    errors = []

    for idx, row in tqdm(subset.iterrows(), total=len(subset), desc="Generating"):
        # Skip if already generated and file exists
        if (
            pd.notna(row.get("filename"))
            and (output_dir / str(row["filename"])).exists()
        ):
            continue

        filename = f"{row['product_id']}.png"
        filepath = output_dir / filename

        image_bytes = generate_image(
            name=row["name"],
            category=row["category"],
            subcategory=row["subcategory"],
            api_key=api_key,
        )

        if image_bytes:
            filepath.write_bytes(image_bytes)
            df.at[idx, "filename"] = filename
        else:
            errors.append(
                {
                    "idx": idx,
                    "name": row["name"],
                    "category": row["category"],
                    "subcategory": row["subcategory"],
                }
            )

        # Intermediate save every 50 images
        if (idx - args.start + 1) % 50 == 0:
            df.to_csv(args.csv, index=False)
            tqdm.write(f"CSV saved at index {idx}")

        # Light pause to avoid rate limiting
        time.sleep(0.3)

    # Final save
    df.to_csv(args.csv, index=False)

    generated = df["filename"].notna().sum()
    print(f"\n{generated}/{len(df)} images generated")
    print(f"Directory: {output_dir.resolve()}")

    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors[:10]:
            print(f"   - [{e['idx']}] {e['name']} ({e['category']}/{e['subcategory']})")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()
