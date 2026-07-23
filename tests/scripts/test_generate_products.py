import importlib.util
import random
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "generate_products.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_products", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gp = _load_module()


@pytest.fixture(autouse=True)
def _seed_random():
    random.seed(42)
    np.random.seed(42)
    yield


class TestTaxonomyData:
    def test_categories_and_subcategories_are_aligned(self):
        assert set(gp.CATEGORIES.keys()) == set(gp.SUB_CATEGORIES.keys())

    def test_weights_sum_to_one(self):
        total_weight = sum(config["weight"] for config in gp.CATEGORIES.values())
        assert total_weight == pytest.approx(1.0)

    def test_price_ranges_are_valid(self):
        for category, config in gp.CATEGORIES.items():
            for subcategory, price_range in config["subcategories"].items():
                low, high = price_range
                assert 0 < low < high, f"{category}/{subcategory}"

    def test_each_category_has_non_empty_unique_subcategories(self):
        for category, sub_categories in gp.SUB_CATEGORIES.items():
            assert len(sub_categories) > 0, category
            assert len(sub_categories) == len(set(sub_categories)), category


class TestPriceBucket:
    @pytest.mark.parametrize(
        "price, expected_bucket",
        [
            (0, 0),
            (24.99, 0),
            (25, 1),
            (74.99, 1),
            (75, 2),
            (199.99, 2),
            (200, 3),
            (499.99, 3),
            (500, 4),
            (10_000, 4),
        ],
    )
    def test_boundaries(self, price, expected_bucket):
        assert gp.price_bucket(price) == expected_bucket


class TestGenerateProduct:
    @pytest.mark.parametrize(
        "category, config", list(gp.CATEGORIES.items()), ids=list(gp.CATEGORIES.keys())
    )
    def test_fields_are_well_formed(self, category, config):
        sub_category = gp.SUB_CATEGORIES[category][0]
        product = gp.generate_product(category, sub_category, config)

        uuid.UUID(product["product_id"])
        assert product["name"] is None
        assert product["filename"] is None

        assert product["category"] == category
        assert product["subcategory"] == sub_category

        low, high = config["subcategories"][sub_category]
        assert low <= product["price"] <= high
        assert product["price_bucket"] == gp.price_bucket(product["price"])

        assert 0.01 <= product["popularity"] <= 0.30

        assert isinstance(product["is_new"], bool)
        assert product["discount_pct"] in {0, 10, 20, 30}

    def test_price_and_popularity_are_clipped_to_config_bounds(self, monkeypatch):
        config = {"subcategories": {"smartphones": (10, 20)}, "base_ctr": 0.5}
        monkeypatch.setattr(np.random, "lognormal", lambda mean, sigma: 1_000_000)
        monkeypatch.setattr(np.random, "normal", lambda loc, scale: 999)

        product = gp.generate_product("electronics", "smartphones", config)

        assert product["price"] == 20.0
        assert product["popularity"] == 0.30

    def test_unique_ids_across_calls(self):
        config = gp.CATEGORIES["books"]
        ids = {
            gp.generate_product("books", "fiction", config)["product_id"]
            for _ in range(50)
        }
        assert len(ids) == 50


class TestGenerateCatalog:
    def test_default_size(self):
        catalog = gp.generate_catalog()
        assert len(catalog) == gp.PRODUCT_TO_GENERATE_COUNT

    @pytest.mark.parametrize("n", [0, 1, 7, 50, 500])
    def test_requested_size_is_respected(self, n):
        catalog = gp.generate_catalog(n)
        assert len(catalog) == n

    def test_products_use_known_categories_and_subcategories(self):
        catalog = gp.generate_catalog(200)
        for product in catalog:
            assert product["category"] in gp.CATEGORIES
            assert product["subcategory"] in gp.SUB_CATEGORIES[product["category"]]

    def test_product_ids_are_unique(self):
        catalog = gp.generate_catalog(300)
        ids = {product["product_id"] for product in catalog}
        assert len(ids) == len(catalog)

    def test_category_distribution_matches_weights_when_no_rounding_needed(self):
        # With n=500, the sum of per-category rounded counts lands exactly
        # on 500, so no random top-up product is added.
        n = 500
        catalog = gp.generate_catalog(n)
        counts = {}
        for product in catalog:
            counts[product["category"]] = counts.get(product["category"], 0) + 1

        for category, config in gp.CATEGORIES.items():
            assert counts[category] == round(n * config["weight"])


class TestMainScript:
    def test_running_as_script_writes_expected_csv(self, tmp_path):
        (tmp_path / "data").mkdir()

        result = subprocess.run(
            [sys.executable, str(MODULE_PATH)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr

        csv_path = tmp_path / "data" / "products.csv"
        assert csv_path.exists()

        df = pd.read_csv(csv_path)
        assert len(df) == gp.PRODUCT_TO_GENERATE_COUNT
        assert set(df["category"].unique()) <= set(gp.CATEGORIES.keys())
