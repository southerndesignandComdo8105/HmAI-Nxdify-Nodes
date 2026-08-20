from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import minimax_autoprompt_nodes as auto


def image_batch(count=1, height=8, width=12):
    values = np.linspace(0.0, 1.0, count * height * width * 3, dtype=np.float32)
    return values.reshape(count, height, width, 3)


class TwoStagePromptTests(unittest.TestCase):
    def test_exact_model_ids(self):
        self.assertEqual(auto.VISUAL_MODEL, "mistralai/mistral-medium-3-5")
        self.assertEqual(
            auto.WRITER_MODELS,
            ("deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-pro"),
        )
        i2v_inputs = auto.MiniMaxH3I2VTwoStagePrompt.INPUT_TYPES()["required"]
        r2v_inputs = auto.MiniMaxH3R2VTwoStagePrompt.INPUT_TYPES()["required"]
        self.assertEqual(i2v_inputs["visual_model"][0], [auto.VISUAL_MODEL])
        self.assertEqual(i2v_inputs["prompt_writer_model"][0], list(auto.WRITER_MODELS))
        self.assertEqual(r2v_inputs["sampled_video_frames"][1]["default"], 16)
        self.assertEqual(r2v_inputs["sampled_video_frames"][1]["min"], 6)
        self.assertEqual(r2v_inputs["sampled_video_frames"][1]["max"], 32)

    def test_i2v_auto_is_image_then_text(self):
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return "FACTUAL ANALYSIS" if len(calls) == 1 else "FINAL I2V PROMPT"

        with patch.object(auto, "_openrouter_completion", side_effect=fake_completion):
            result = auto.MiniMaxH3I2VTwoStagePrompt().generate(
                image_batch(), "auto", auto.VISUAL_MODEL, auto.WRITER_MODELS[0],
                "turn toward the window", "unused", "test-key", 1280, 720, 5.0,
            )

        self.assertEqual(result[:2], ("FACTUAL ANALYSIS", "FINAL I2V PROMPT"))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["model"], auto.VISUAL_MODEL)
        visual_content = calls[0]["messages"][1]["content"]
        self.assertTrue(any(part["type"] == "image_url" for part in visual_content))
        self.assertEqual(calls[1]["model"], auto.WRITER_MODELS[0])
        self.assertIsInstance(calls[1]["messages"][1]["content"], str)
        self.assertIn("FACTUAL ANALYSIS", calls[1]["messages"][1]["content"])
        self.assertNotIn("data:image", calls[1]["messages"][1]["content"])

    def test_i2v_manual_makes_zero_calls(self):
        with patch.object(auto, "_openrouter_completion") as completion:
            result = auto.MiniMaxH3I2VTwoStagePrompt().generate(
                None, "manual", auto.VISUAL_MODEL, auto.WRITER_MODELS[0], "", "MANUAL I2V",
                "", 1280, 720, 5.0,
            )
        completion.assert_not_called()
        self.assertEqual(result[1], "MANUAL I2V")

    def test_r2v_auto_samples_full_video_and_sends_text_to_writer(self):
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return "MOTION ANALYSIS" if len(calls) == 1 else "FINAL R2V PROMPT"

        with (
            patch.object(auto, "_openrouter_completion", side_effect=fake_completion),
            patch.object(auto, "_r2v_writer_instructions", return_value="REF2VA AUTHORITY"),
        ):
            result = auto.MiniMaxH3R2VTwoStagePrompt().generate(
                "auto", auto.VISUAL_MODEL, auto.WRITER_MODELS[0], "continue the movement",
                "unused", 16, "test-key", 1280, 720, 8.0, "standard",
                video_1=image_batch(48),
            )

        self.assertEqual(result[:2], ("MOTION ANALYSIS", "FINAL R2V PROMPT"))
        parts = calls[0]["messages"][1]["content"]
        image_parts = [part for part in parts if part["type"] == "image_url"]
        self.assertEqual(len(image_parts), 16)
        labels = "\n".join(part.get("text", "") for part in parts if part["type"] == "text")
        self.assertIn("<Video 1> Frame 01/16", labels)
        self.assertIn("chronological position 1.000", labels)
        self.assertNotIn("video_url", repr(calls[0]["messages"]))
        writer_content = calls[1]["messages"][1]["content"]
        self.assertIsInstance(writer_content, str)
        self.assertIn("MOTION ANALYSIS", writer_content)
        self.assertNotIn("data:image", writer_content)

    def test_r2v_combined_references_preserve_tags_and_audio_reservations(self):
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return "ANALYSIS" if len(calls) == 1 else "PROMPT"

        with (
            patch.object(auto, "_openrouter_completion", side_effect=fake_completion),
            patch.object(auto, "_r2v_writer_instructions", return_value="AUTHORITY"),
        ):
            auto.MiniMaxH3R2VTwoStagePrompt().generate(
                "auto", auto.VISUAL_MODEL, auto.WRITER_MODELS[1], "replace the object",
                "unused", 6, "test-key", 1080, 1920, 8.0, "replacement",
                image_1=image_batch(), video_1=image_batch(12), video_audio_1={"audio": True},
                audio_1={"audio": True},
            )

        writer = calls[1]["messages"][1]["content"]
        self.assertIn("<Picture 1>", writer)
        self.assertIn("<Video 1> <Audio 1>", writer)
        self.assertIn("<Audio 2>", writer)
        self.assertNotIn("<Picture 2>", writer)
        visual_parts = calls[0]["messages"][1]["content"]
        self.assertEqual(sum(part["type"] == "image_url" for part in visual_parts), 7)

    def test_r2v_manual_makes_zero_calls_without_references_or_key(self):
        with patch.object(auto, "_openrouter_completion") as completion:
            result = auto.MiniMaxH3R2VTwoStagePrompt().generate(
                "manual", auto.VISUAL_MODEL, auto.WRITER_MODELS[0], "", "MANUAL R2V",
                16, "", 1280, 720, 8.0, "auto",
            )
        completion.assert_not_called()
        self.assertEqual(result[1], "MANUAL R2V")


class WorkflowTests(unittest.TestCase):
    def test_workflow_links_and_prompt_targets(self):
        for path in sorted((ROOT / "workflows").glob("*_AutoPrompt_*.json")):
            graph = json.loads(path.read_text(encoding="utf-8"))
            nodes = {item["id"]: item for item in graph["nodes"]}
            link_ids = set()
            for link in graph["links"]:
                link_id, origin, origin_slot, target, target_slot, _ = link
                self.assertNotIn(link_id, link_ids, path.name)
                link_ids.add(link_id)
                self.assertIn(origin, nodes, path.name)
                self.assertIn(target, nodes, path.name)
                self.assertLess(origin_slot, len(nodes[origin].get("outputs", [])), path.name)
                self.assertLess(target_slot, len(nodes[target].get("inputs", [])), path.name)

            text = path.read_text(encoding="utf-8")
            self.assertNotIn("google/gemini", text)
            self.assertNotIn("video_url", text)
            if "I2V" in path.name:
                prompt_node = next(
                    item for item in graph["nodes"]
                    if item["type"] == "MiniMaxH3I2VTwoStagePrompt"
                )
                if "Integrated" in path.name:
                    core = next(item for item in graph["nodes"] if item["type"] == "MiniMaxH3ImageToVideo")
                    prompt_link = core["inputs"][4]["link"]
                    self.assertTrue(any(link[0] == prompt_link and link[1] == prompt_node["id"] and link[2] == 1 for link in graph["links"]))
            else:
                refpack = next(item for item in graph["nodes"] if item["type"] == "MiniMaxH3ReferencePack")
                self.assertEqual(refpack["widgets_values"][3], "none")
                prompt_node = next(item for item in graph["nodes"] if item["type"] == "MiniMaxH3R2VTwoStagePrompt")
                core = next(item for item in graph["nodes"] if item["type"] == "MiniMaxH3ReferenceToVideo")
                prompt_link = core["inputs"][21]["link"]
                self.assertTrue(any(link[0] == prompt_link and link[1] == prompt_node["id"] and link[2] == 1 for link in graph["links"]))

    def test_integrated_generation_configuration_is_preserved(self):
        i2v = json.loads(
            (ROOT / "workflows" / "I2V_AutoPrompt_Integrated.json").read_text(encoding="utf-8")
        )
        i2v_nodes = {item["type"]: item for item in i2v["nodes"]}
        self.assertEqual(
            i2v_nodes["UNETLoader"]["widgets_values"],
            ["minimax_h3_fl2va_pruned_int8_convrot.safetensors", "default"],
        )
        self.assertEqual(
            i2v_nodes["LoraLoaderModelOnly"]["widgets_values"],
            ["minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors", 0.5],
        )
        self.assertEqual(i2v_nodes["BetaSamplingScheduler"]["widgets_values"], [8, 0.79, 0.5])
        self.assertEqual(i2v_nodes["MiniMaxH3ImageToVideo"]["widgets_values"][1:], [1344, 768, 73])

        r2v = json.loads(
            (ROOT / "workflows" / "R2V_AutoPrompt_Integrated.json").read_text(encoding="utf-8")
        )
        r2v_nodes = {item["type"]: item for item in r2v["nodes"]}
        self.assertEqual(
            r2v_nodes["UNETLoader"]["widgets_values"],
            ["minimax_h3_ref2va_int8_convrot.safetensors", "default"],
        )
        self.assertEqual(
            r2v_nodes["LoraLoaderModelOnly"]["widgets_values"],
            ["minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", 0.85],
        )
        self.assertEqual(r2v_nodes["BetaSamplingScheduler"]["widgets_values"], [4, 0.6, 0.6])
        self.assertEqual(
            r2v_nodes["MiniMaxH3ReferenceToVideo"]["widgets_values"][1:],
            [1344, 768, 124, "max"],
        )


if __name__ == "__main__":
    unittest.main()
