"""Seeded Plackett-Luce-style race order sampling with explicit DNF handling."""

from typing import Mapping

import numpy as np


def plackett_luce_probabilities(
    strengths: Mapping[str, float], eligible_drivers: list[str] | None = None
) -> dict[str, float]:
    """Return normalized strengths for one Plackett-Luce selection step."""
    selected = list(strengths) if eligible_drivers is None else eligible_drivers
    if not selected:
        return {}
    if len(set(selected)) != len(selected) or set(selected) - set(strengths):
        raise ValueError("Eligible drivers must be unique keys in strengths")
    weights = np.asarray([strengths[driver] for driver in selected], dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Every eligible driver strength must be finite and greater than zero")
    probabilities = weights / weights.sum()
    return dict(zip(selected, probabilities.tolist()))


class RaceSimulator:
    """Generate every driver's result once; place sampled DNFs after finishers.

    The model treats supplied strengths as relative selection weights. DNF
    status is sampled independently per driver, and DNF ordering is randomized
    because lap-completion information is not part of this input contract.
    """

    def simulate(
        self,
        strengths: Mapping[str, float],
        dnf_probabilities: Mapping[str, float] | None = None,
        random_seed: int | None = None,
    ) -> list[dict[str, str | int | bool | None]]:
        if not strengths:
            raise ValueError("At least one eligible driver is required")
        drivers = list(strengths)
        if len(set(drivers)) != len(drivers) or any(not driver for driver in drivers):
            raise ValueError("Driver identifiers must be unique and non-empty")
        plackett_luce_probabilities(strengths)
        dnf_probabilities = dnf_probabilities or {}
        unknown = set(dnf_probabilities) - set(drivers)
        if unknown:
            raise ValueError(f"DNF probabilities contain unknown drivers: {sorted(unknown)}")
        normalized_dnf: dict[str, float] = {}
        for driver in drivers:
            probability = float(dnf_probabilities.get(driver, 0.0))
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"Invalid DNF probability for {driver}: {probability}")
            normalized_dnf[driver] = probability

        rng = np.random.default_rng(random_seed)
        dnf_mask = np.asarray(
            [rng.random() < normalized_dnf[driver] for driver in drivers],
            dtype=bool,
        )
        weights = np.asarray([strengths[driver] for driver in drivers], dtype=float)
        finishers = np.asarray(drivers, dtype=object)[~dnf_mask]
        finisher_weights = weights[~dnf_mask]
        exponential_clocks = rng.exponential(
            scale=1.0 / finisher_weights,
            size=len(finisher_weights),
        )
        finish_order = finishers[np.argsort(exponential_clocks)].tolist()
        dnf_order = np.asarray(drivers, dtype=object)[dnf_mask].tolist()
        dnf_drivers = set(dnf_order)
        if dnf_order:
            rng.shuffle(dnf_order)
        complete_order = finish_order + dnf_order
        return [
            {
                "driver_id": driver,
                "position": position,
                "classified_position": position if driver not in dnf_drivers else None,
                "status": "DNF" if driver in dnf_drivers else "Finished",
                "classified": driver not in dnf_drivers,
            }
            for position, driver in enumerate(complete_order, start=1)
        ]
