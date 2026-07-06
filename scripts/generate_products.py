# generate_products.py

import uuid
import random
import numpy as np
import pandas as pd


PRODUCT_TO_GENERATE_COUNT = 500  # Number of generated products


# Taxonomy
# TODO
# Price range must be change from price/cat to price /subcat
# food_gourmet cat changes to food -> Update the dataset


# price_range = (min_price, max_price), base_ctr = base click-through rate, weight = proportion of products in this category
CATEGORIES = {
    "electronics": {
        "price_range": (50, 1200),
        "base_ctr": 0.08,
        "weight": 0.18,
    },
    "fashion": {"price_range": (15, 300), "base_ctr": 0.12, "weight": 0.22},
    "home_deco": {"price_range": (20, 500), "base_ctr": 0.07, "weight": 0.14},
    "sports": {"price_range": (25, 600), "base_ctr": 0.09, "weight": 0.12},
    "beauty": {"price_range": (10, 150), "base_ctr": 0.14, "weight": 0.10},
    "books": {"price_range": (8, 45), "base_ctr": 0.06, "weight": 0.09},
    "food": {"price_range": (5, 80), "base_ctr": 0.11, "weight": 0.08},
    "toys_games": {"price_range": (12, 200), "base_ctr": 0.10, "weight": 0.07},
}


SUB_CATEGORIES = {
    "electronics": ["smartphones", "laptops", "tablets", "headphones", "cameras"],
    "fashion": ["clothing", "shoes", "accessories", "jewelry"],
    "home_deco": ["furniture", "lighting", "decorative_items"],
    "sports": ["fitness_equipment", "outdoor_gear", "sportswear"],
    "beauty": ["skincare", "makeup", "haircare"],
    "books": ["fiction", "non-fiction", "children_books"],
    "food": ["snacks", "beverages", "gourmet_foods"],
    "toys_games": ["board_games", "action_figures", "puzzles"],
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
    low, high = config["price_range"]
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
