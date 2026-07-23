# generate_products.py

import uuid
import random
import numpy as np
import pandas as pd


PRODUCT_TO_GENERATE_COUNT = 500  # Number of generated products


# Taxonomy
# base_ctr = base click-through rate, weight = proportion of products in this category
CATEGORIES = {
    "electronics": {
        "base_ctr": 0.08,
        "weight": 0.18,
        "subcategories": {
            "smartphones": (150, 1_200),
            "laptops": (400, 2_500),
            "tablets": (120, 1_400),
            "headphones": (25, 600),
            "cameras": (150, 2_000),
        },
    },
    "fashion": {
        "base_ctr": 0.12,
        "weight": 0.22,
        "subcategories": {
            "clothing": (15, 250),
            "shoes": (30, 350),
            "accessories": (10, 150),
            "jewelry": (20, 800),
        },
    },
    "home_deco": {
        "base_ctr": 0.07,
        "weight": 0.14,
        "subcategories": {
            "furniture": (80, 2_000),
            "lighting": (25, 500),
            "decorative_items": (10, 300),
        },
    },
    "sports": {
        "base_ctr": 0.09,
        "weight": 0.12,
        "subcategories": {
            "fitness_equipment": (40, 1_200),
            "outdoor_gear": (25, 800),
            "sportswear": (15, 250),
        },
    },
    "beauty": {
        "base_ctr": 0.14,
        "weight": 0.10,
        "subcategories": {
            "skincare": (10, 180),
            "makeup": (8, 120),
            "haircare": (8, 100),
        },
    },
    "books": {
        "base_ctr": 0.06,
        "weight": 0.09,
        "subcategories": {
            "fiction": (8, 35),
            "non-fiction": (12, 60),
            "children_books": (6, 30),
        },
    },
    "food": {
        "base_ctr": 0.11,
        "weight": 0.08,
        "subcategories": {
            "snacks": (2, 20),
            "beverages": (3, 35),
            "gourmet_foods": (8, 100),
        },
    },
    "toys_games": {
        "base_ctr": 0.10,
        "weight": 0.07,
        "subcategories": {
            "board_games": (15, 100),
            "action_figures": (10, 150),
            "puzzles": (8, 80),
        },
    },
}


SUB_CATEGORIES = {
    category: list(config["subcategories"]) for category, config in CATEGORIES.items()
}


def price_bucket(price: float) -> int:
    """
    Discretize the price into a bucket for LinUCB (categorical feature).
    """
    if price < 25:
        return 0  # budget
    if price < 75:
        return 1  # accessible
    if price < 200:
        return 2  # mid-range
    if price < 500:
        return 3  # premium
    return 4  # luxury


def generate_product(category: str, sub_category: str, config: dict) -> dict:
    low, high = config["subcategories"][sub_category]
    # Log-normal distribution: realistic, long tail towards higher prices
    price = round(np.random.lognormal(mean=np.log((low + high) / 2), sigma=0.4), 2)
    price = float(np.clip(price, low, high))

    # Popularity correlated with base CTR + noise
    popularity = float(np.clip(np.random.normal(config["base_ctr"], 0.03), 0.01, 0.30))

    return {
        "product_id": str(uuid.uuid4()),
        "name": None,
        "category": category,
        "subcategory": sub_category,
        "price": price,
        "price_bucket": price_bucket(price),
        "popularity": round(popularity, 4),
        "is_new": random.random() < 0.15,  # 15% new arrivals
        "discount_pct": random.choice([0, 0, 0, 10, 20, 30]),  # 0 is the majority
        "filename": None,
    }


def generate_catalog(n: int = 500) -> list[dict]:
    """
    Generate a product catalog of n products, distributed across categories
    according to their weight and randomly assigned a subcategory.
    """
    products = []
    for category, config in CATEGORIES.items():
        count = round(n * config["weight"])
        for i, _ in enumerate(range(count)):
            sub_category = SUB_CATEGORIES[category][i % len(SUB_CATEGORIES[category])]
            products.append(generate_product(category, sub_category, config))

    # Adjustment if the total != n because of rounding
    while len(products) < n:
        cat = random.choice(list(CATEGORIES.keys()))
        sub_category = random.choice(SUB_CATEGORIES[cat])
        products.append(generate_product(cat, sub_category, CATEGORIES[cat]))

    return products[:n]


if __name__ == "__main__":
    catalog = generate_catalog(PRODUCT_TO_GENERATE_COUNT)
    df = pd.DataFrame(catalog)

    df.to_csv("data/products.csv", index=False)

    print(df.groupby("category")[["price", "popularity"]].mean().round(3))
