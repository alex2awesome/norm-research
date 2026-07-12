"""Regression tests for audited capability-v1 counterexamples."""

import unittest
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ops_capability_v2 as ops


class CapabilityV2Tests(unittest.TestCase):
    def test_missing_year_is_frozen(self):
        rows = ops.date_chain("The notice arrived April 2.")
        self.assertEqual(rows[0]["date"], "2000-04-02")
        self.assertEqual(rows[0]["parse_status"], "VALID")
        self.assertTrue(rows[0]["checkable"])

    def test_invalid_calendar_date_is_explicit_not_dropped_or_clamped(self):
        rows = ops.date_chain(
            "The notice arrived April 31, 1982. It was filed May 3, 1982."
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["text"], "April 31, 1982")
        self.assertIsNone(rows[0]["date"])
        self.assertEqual(rows[0]["parse_status"], "INVALID")
        self.assertFalse(rows[0]["checkable"])
        # Invalid evidence cannot silently become the interval anchor for the next row.
        self.assertEqual(rows[1]["date"], "1982-05-03")
        self.assertIsNone(rows[1]["days_since_prev"])

    def test_negative_deadline_gap_fails(self):
        self.assertFalse(ops.deadline_satisfied("2020-02-01", "2020-01-01", 90))

    def test_number_direction_is_not_discarded(self):
        rows = ops.number_consistency("It decreased from 100 to 50, a 50% increase.")
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["direction_consistent"])
        self.assertFalse(rows[0]["consistent"])

    def test_p_bounds_are_not_point_decisions(self):
        rows = ops.stat_consistency("The estimate was z = 2.58, p > .001.")
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["decision_inconsistent"])
        rows = ops.stat_consistency("The estimate was z = 2.58, p < .10.")
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["decision_inconsistent"])

    def test_adjacent_variation_is_not_refrain(self):
        text = "We will win today. We really will win today. A final sentence."
        rows = ops.is_refrain(text)
        for row in rows:
            if row["occurrences"][:2] == [0, 1]:
                self.assertFalse(row["is_refrain"])

    def test_one_word_refrain_can_be_craft(self):
        rows = ops.is_refrain(
            "Never. The river climbed and swallowed the road. Never."
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["occurrences"], [0, 2])
        self.assertTrue(rows[0]["has_intervening_sentence"])
        self.assertTrue(rows[0]["is_refrain"])

    def test_one_word_adjacent_repetition_is_not_craft(self):
        rows = ops.is_refrain("Never. Never. The river climbed.")
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["has_intervening_sentence"])
        self.assertFalse(rows[0]["is_refrain"])

    def test_historical_two_word_refrain_probe_is_detected_locally(self):
        text = (
            "The outer wall fell. They fought. The docks burned. They fought. "
            "By the time only the harbor was left, they were still fighting."
        )
        rows = ops.is_refrain(text)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sentence"], "They fought.")
        self.assertEqual(rows[0]["occurrences"], [1, 3])
        self.assertTrue(rows[0]["is_refrain"])

    def test_invalid_and_ambiguous_positions_abstain(self):
        text = "Same line. Middle line. Same line."
        self.assertIsNone(ops.discourse_position(text, "Same line."))
        self.assertIsNone(ops.discourse_position(text, (-1, 2)))
        self.assertIsNone(ops.discourse_position(text, (999, 1000)))

    def test_repeated_and_capped_attribution_spans_abstain(self):
        text = "Repeated claim. Repeated claim."
        self.assertIsNone(ops.self_attributed(text, "Repeated claim."))
        long_text = "x" * 8100 + " unique tail"
        self.assertIsNone(ops.self_attributed(long_text, "unique tail"))

    def test_multi_org_and_missing_issuer_abstain(self):
        text = (
            "Apple announced that sales grew. Google said that demand would grow. "
            "Microsoft noted that prices fell."
        )
        rows = ops.attributions(text)
        self.assertTrue(rows)
        self.assertTrue(all(r["speaker_is_first_person_org"] is None for r in rows))
        text = '"We are proud," said Maria Chen at Acme Biotech.'
        rows = ops.attributions(text)
        self.assertTrue(rows)
        self.assertIsNone(rows[0]["speaker_is_first_person_org"])

    def test_reporting_conjunct_inherits_shared_subject(self):
        text = (
            "Wren starved the city, let drought do its work, and told the council "
            "exactly what she'd done."
        )
        rows = ops.attributions(text)
        inherited = [
            r for r in rows if r.get("attribution_mode") == "reporting_verb_shared_subject"
        ]
        self.assertEqual(len(inherited), 1)
        self.assertEqual(inherited[0]["verb"], "tell")
        self.assertEqual(inherited[0]["speaker_span"], "Wren")
        self.assertIn("what she'd done", inherited[0]["span"])

    def test_named_action_beat_anchors_following_quote(self):
        text = (
            "Bren the smith stepped forward first. "
            "'Double it, and send the coins to the east gate.'"
        )
        rows = ops.attributions(text)
        beats = [
            r for r in rows if r.get("attribution_mode") == "adjacent_named_action_beat"
        ]
        self.assertEqual(len(beats), 1)
        self.assertEqual(beats[0]["speaker_span"], "Bren the smith")
        self.assertEqual(
            beats[0]["span"], "Double it, and send the coins to the east gate."
        )
        self.assertEqual(
            beats[0]["attribution_status"], "bounded_syntactic_association"
        )
        self.assertIsNone(beats[0]["speaker_is_first_person_org"])

    def test_action_beat_requires_explicit_named_subject(self):
        rows = ops.attributions("She stepped forward first. 'Double it.'")
        self.assertFalse(
            any(r.get("attribution_mode") == "adjacent_named_action_beat" for r in rows)
        )

    def test_historical_multi_actor_action_beats_cross_sentence_quotes(self):
        text = (
            "'The bridge toll doubles this spring,' the reeve announced. 'Speak now.'\n\n"
            "Bren the smith stepped forward first. 'Double it and I lose every customer "
            "on the far bank. Cap it, don't raise it.'\n\n"
            "Old Yara leaned on her cane. 'I don't care about the toll. I want the ferry "
            "running again — my knees can't take the bridge steps anymore.'\n\n"
            "Young Cass didn't wait to be called on. 'None of that matters if the bridge "
            "collapses first. Fund the repair, then argue about the toll.'"
        )
        beats = [
            r
            for r in ops.attributions(text)
            if r.get("attribution_mode") == "adjacent_named_action_beat"
        ]
        self.assertEqual(
            [r["speaker_span"] for r in beats],
            ["Bren the smith", "Old Yara", "Young Cass"],
        )
        self.assertIn("don't raise it", beats[0]["span"])
        self.assertIn("my knees can't", beats[1]["span"])
        self.assertIn("Fund the repair", beats[2]["span"])


if __name__ == "__main__":
    unittest.main()
