"""
LinUCB - Bandit Model
===============================================
Choix d'architecture V1 :
  - 8 bras (1 par catégorie produit)
  - features x dim=13 : affinity×8 + device one-hot (3) + hour sin/cos (2)
  - alpha exposé : contrôle exploration vs exploitation
  - Couche XAI intégrée : top-3 features + score UCB par bras

"""

import numpy as np
import json
from dataclasses import dataclass, field
from typing import Optional



# TODO: Swap categories mode to product mode

CATEGORIES = [
    "electronics", "fashion", "home_deco", "sports",
    "beauty", "books", "food_gourmet", "toys_games"
]
DEVICES = ["mobile", "desktop", "tablet"]
N_ARMS   = len(CATEGORIES)    # Change this to len(PRODUCTS) with a product list
DIM      = 13                 # Change this based on the dimensions (affinity×8 + device×3 + sin(hour) + cos(hour))

FEATURE_NAMES = (
    [f"affinity_{c}" for c in CATEGORIES]   # 0-7
    + [f"device_{d}" for d in DEVICES]      # 8-10
    + ["hour_sin", "hour_cos"]              # 11-12
)


# Context Encoder
def encode_context(
    affinities: dict[str, float],
    device: str,
    hour: int,
) -> np.ndarray:
    """
    Build x ∈ R^13 from visitor informations

    Args:
        affinities : dict {categorie: score [0,1]} — Must be changed using products
        device     : "mobile" | "desktop" | "tablet"
        hour       :  0-23 Local hour

    Returns:
        np.ndarray shape (13,) (normalized)
    """
    # Affinities
    aff = np.array([affinities.get(cat, 0.0) for cat in CATEGORIES], dtype=np.float64)

    # Device
    dev = np.zeros(3, dtype=np.float64)
    if device in DEVICES:
        dev[DEVICES.index(device)] = 1.0

    # 3. Hours (sin/cos) to catch cycles
    angle = 2 * np.pi * hour / 24
    hour_encoded = np.array([np.sin(angle), np.cos(angle)], dtype=np.float64)

    x = np.concatenate([aff, dev, hour_encoded])
    assert x.shape == (DIM,), f"Dimension : {x.shape}"
    return x


# LinUCB Agent

@dataclass
class LinUCBAgent:
    """
    LinUCB Agent.

    For each arm called a :
        θ_a = A_a^{-1} b_a
        UCB_a = θ_a · x + alpha * sqrt(x^T A_a^{-1} x)
              = expected_reward + exploration_bonus

    Attributes:
        alpha      : exploration parameter (>0).
        n_arms     : (n categories)
        dim        : context vector dimension
        A          : matrix list (dim×dim) init to I
        b          : Vector list (dim,), init to 0
        n_pulls    : compteur de tirages par bras
        total_reward : cumulated reward / arm
    """
    alpha: float = 1.0
    n_arms: int  = N_ARMS
    dim: int     = DIM

    # __post_init__ parameters
    A: list          = field(default_factory=list, repr=False)
    b: list          = field(default_factory=list, repr=False)
    n_pulls: list    = field(default_factory=list)
    total_reward: list = field(default_factory=list)

    def __post_init__(self):
        if not self.A:
            self.A = [np.eye(self.dim) for _ in range(self.n_arms)]
            self.b = [np.zeros(self.dim) for _ in range(self.n_arms)]
            self.n_pulls   = [0] * self.n_arms
            self.total_reward = [0.0] * self.n_arms


    def select_arm(self, x: np.ndarray) -> tuple[int, dict]:
        """
        UCB : Select the arm wit hthe best UCB score.

        Returns:
            (arm_idx, xai_payload), xai_payload contains explicability informations returned to users
        """
        scores = self._compute_all_scores(x)
        best_arm = int(np.argmax(scores["ucb"]))
        xai = self._build_xai(x, best_arm, scores)
        return best_arm, xai

    def select_top_k(self, x: np.ndarray, k: int = 3) -> tuple[list[int], dict]:
        """
        Select top k arms to recommend products
        """
        scores = self._compute_all_scores(x)
        top_k = list(np.argsort(scores["ucb"])[::-1][:k])
        xai = self._build_xai(x, top_k[0], scores)
        return top_k, xai


    def update(self, arm: int, x: np.ndarray, reward: float) -> None:
        """
        Update A_arm & b_arm

        Formula :
            A_a ← A_a + x x^T
            b_a ← b_a + r * x
        """
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self.n_pulls[arm]    += 1
        self.total_reward[arm] += reward


    def _compute_all_scores(self, x: np.ndarray) -> dict:
        """
        Compute expected_reward, exploration_bonus & UCB for all arms.
        """
        expected = np.zeros(self.n_arms)
        bonus    = np.zeros(self.n_arms)
        theta    = []

        for a in range(self.n_arms):
            A_inv    = np.linalg.inv(self.A[a])
            theta_a  = A_inv @ self.b[a]
            theta.append(theta_a)
            expected[a] = theta_a @ x
            bonus[a]    = self.alpha * np.sqrt(x @ A_inv @ x)

        return {
            "expected": expected,
            "bonus":    bonus,
            "ucb":      expected + bonus,
            "theta":    theta,
        }



    def _build_xai(self, x: np.ndarray, best_arm: int, scores: dict) -> dict:
        """
        Build the UI payload

        Content :
          - chosen_category    : Recommended cat
          - ucb_scores         : UCB score for each cat/product
          - expected_reward
          - exploration_bonus
          - top3_features
          - n_interactions     : Total updates nb
        """
        theta_a = scores["theta"][best_arm]

        # Feature contribution = |θ_i * x_i|
        contributions = np.abs(theta_a * x)
        top3_idx = np.argsort(contributions)[::-1][:3]

        top3_features = [
            {
                "feature":      FEATURE_NAMES[i],
                "weight":       round(float(theta_a[i]), 4),
                "value":        round(float(x[i]), 4),
                "contribution": round(float(contributions[i]), 4),
            }
            for i in top3_idx
        ]

        ucb_scores = {
            CATEGORIES[a]: {
                "ucb":      round(float(scores["ucb"][a]), 4),
                "expected": round(float(scores["expected"][a]), 4),
                "bonus":    round(float(scores["bonus"][a]), 4),
                "n_pulls":  self.n_pulls[a],
            }
            for a in range(self.n_arms)
        }

        return {
            "chosen_category":   CATEGORIES[best_arm],
            "ucb_scores":        ucb_scores,
            "expected_reward":   round(float(scores["expected"][best_arm]), 4),
            "exploration_bonus": round(float(scores["bonus"][best_arm]), 4),
            "top3_features":     top3_features,
            "n_interactions":    sum(self.n_pulls),
            "alpha":             self.alpha,
        }



    def to_dict(self) -> dict:
        """
        Agent serialization to JSON format.
        """
        return {
            "alpha":        self.alpha,
            "n_arms":       self.n_arms,
            "dim":          self.dim,
            "A":            [a.tolist() for a in self.A],
            "b":            [b.tolist() for b in self.b],
            "n_pulls":      self.n_pulls,
            "total_reward": self.total_reward,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LinUCBAgent":
        """
        Load the agent form a dcit (JSON source)
        """
        agent = cls(
            alpha=data["alpha"],
            n_arms=data["n_arms"],
            dim=data["dim"],
            A=[np.array(a) for a in data["A"]],
            b=[np.array(b) for b in data["b"]],
            n_pulls=data["n_pulls"],
            total_reward=data["total_reward"],
        )
        return agent


    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)


    @classmethod
    def load(cls, path: str) -> "LinUCBAgent":
        with open(path) as f:
            return cls.from_dict(json.load(f))


    def convergence_stats(self) -> dict:
        """
        Stat summary for each arm
        """
        return {
            CATEGORIES[a]: {
                "n_pulls":     self.n_pulls[a],
                "avg_reward":  round(
                    self.total_reward[a] / max(self.n_pulls[a], 1), 4
                ),
                "total_reward": round(self.total_reward[a], 4),
            }
            for a in range(self.n_arms)
        }