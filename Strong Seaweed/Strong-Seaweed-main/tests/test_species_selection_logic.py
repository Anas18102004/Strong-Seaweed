import unittest

import species_selection_logic as logic


def row(
    species_id: str,
    prob: float,
    rank: float,
    margin: float,
    actionability: str,
    reason: str = "screening_positive",
    prior: float = 0.5,
) -> dict:
    return {
        "speciesId": species_id,
        "ready": True,
        "probabilityPercent": prob,
        "rankScore": rank,
        "marginToThresholdPercent": margin,
        "actionability": actionability,
        "reason": reason,
        "priorSupportScore": prior,
    }


class SpeciesSelectionLogicTests(unittest.TestCase):
    def test_select_best_species_prefers_rank_not_list_order(self):
        a = row("species_a", 100.0, 0.80, 10.0, "recommended")
        b = row("species_b", 100.0, 0.90, 10.0, "recommended")
        best, reason = logic.select_best_species([a, b], [a, b], None, sort_by_rank_fn=logic.sort_by_rank)
        self.assertEqual(best["speciesId"], "species_b")
        self.assertEqual(reason, "ranked_eligible_positive")

    def test_tie_guardrail_can_pick_non_first_and_force_pilot_only(self):
        top1 = row("species_a", 100.0, 0.82, 12.0, "recommended")
        top2 = row("species_b", 100.0, 0.92, 12.0, "recommended")
        diag = {
            "tieDetected": False,
            "tieGapPercent": None,
            "tieCandidates": [],
            "selectionReason": "ranked_eligible_positive",
            "rankScore": None,
            "tieResolved": False,
        }
        warnings = []
        best, forced = logic.apply_tie_guardrail(
            top1,
            [top1, top2],
            diag,
            warnings,
            tie_gap_percent=1.0,
            tie_force_pilot_only=True,
            default_prior_support=0.5,
        )
        self.assertEqual(best["speciesId"], "species_b")
        self.assertEqual(best["actionability"], "test_pilot_only")
        self.assertTrue(forced)
        self.assertTrue(diag["tieDetected"])
        self.assertIn("best_species_tie_detected", warnings)
        self.assertIn("best_species_tie_forced_pilot_only", warnings)

    def test_non_tie_keeps_best(self):
        top1 = row("species_a", 95.0, 0.90, 14.0, "recommended")
        top2 = row("species_b", 90.0, 0.91, 14.0, "recommended")
        diag = {
            "tieDetected": False,
            "tieGapPercent": None,
            "tieCandidates": [],
            "selectionReason": "ranked_eligible_positive",
            "rankScore": None,
            "tieResolved": False,
        }
        warnings = []
        best, forced = logic.apply_tie_guardrail(
            top1,
            [top1, top2],
            diag,
            warnings,
            tie_gap_percent=1.0,
            tie_force_pilot_only=True,
            default_prior_support=0.5,
        )
        self.assertEqual(best["speciesId"], "species_a")
        self.assertFalse(forced)
        self.assertFalse(diag["tieDetected"])
        self.assertEqual(warnings, [])

    def test_env_support_returns_neutral_without_features(self):
        support = logic.species_env_support("kappaphycus_alvarezii", None)
        self.assertAlmostEqual(support, 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
