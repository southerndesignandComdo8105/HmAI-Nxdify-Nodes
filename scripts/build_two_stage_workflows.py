#!/usr/bin/env python3
"""Replace the legacy single-stage prompt branches in the shipped workflows."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"
VISUAL_MODEL = "mistralai/mistral-medium-3-5"
WRITER_MODEL = "deepseek/deepseek-v4-flash"
I2V_MANUAL = (
    "integrated_multimodal_description: [Shot 1] Preserve the start image and "
    "describe the intended motion.\n\noverall_soundscape: Natural synchronized "
    "ambience.\n\nnon_diegetic_music: N/A"
)
R2V_MANUAL = (
    "subject_definitions:\n\nsummary:\n[pure generation] Replace this placeholder "
    "with a complete manual Ref2VA prompt.\n\nretention_analysis:\n\n"
    "detailed_description:\n[Shot 1] Describe the complete shot.\n\n"
    "overall_soundscape:\nNatural synchronized ambience.\n\nnon_diegetic_music:\nN/A"
)


def load(name: str) -> dict:
    return json.loads((WORKFLOWS / name).read_text(encoding="utf-8"))


def save(name: str, graph: dict) -> None:
    (WORKFLOWS / name).write_text(
        json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def node(graph: dict, node_id: int) -> dict:
    return next(item for item in graph["nodes"] if item["id"] == node_id)


def remove_nodes(graph: dict, node_ids: set[int]) -> None:
    graph["nodes"] = [item for item in graph["nodes"] if item["id"] not in node_ids]
    graph["links"] = [
        link for link in graph["links"] if link[1] not in node_ids and link[3] not in node_ids
    ]


def remove_links(graph: dict, predicate) -> None:
    graph["links"] = [link for link in graph["links"] if not predicate(link)]


def add_link(
    graph: dict, origin_id: int, origin_slot: int, target_id: int, target_slot: int,
    data_type: str,
) -> int:
    graph["last_link_id"] = max(
        graph.get("last_link_id", 0), max((link[0] for link in graph["links"]), default=0)
    ) + 1
    link_id = graph["last_link_id"]
    graph["links"].append(
        [link_id, origin_id, origin_slot, target_id, target_slot, data_type]
    )
    return link_id


def normalize_links(graph: dict) -> None:
    by_id = {item["id"]: item for item in graph["nodes"]}
    for item in graph["nodes"]:
        for input_spec in item.get("inputs", []):
            input_spec["link"] = None
        for output in item.get("outputs", []):
            output["links"] = []

    for link_id, origin_id, origin_slot, target_id, target_slot, _ in graph["links"]:
        origin = by_id[origin_id]
        target = by_id[target_id]
        origin["outputs"][origin_slot]["links"].append(link_id)
        target["inputs"][target_slot]["link"] = link_id


def display_node(node_id: int, title: str, pos: list[float]) -> dict:
    return {
        "id": node_id,
        "type": "Display Any (rgthree)",
        "pos": pos,
        "size": [360, 150],
        "flags": {},
        "order": 99,
        "mode": 0,
        "inputs": [{"name": "source", "type": "*", "link": None}],
        "outputs": [],
        "title": title,
        "properties": {"Node name for S&R": "Display Any (rgthree)"},
        "widgets_values": [""],
    }


def i2v_prompt_node(node_id: int, pos: list[float]) -> dict:
    return {
        "id": node_id,
        "type": "MiniMaxH3I2VTwoStagePrompt",
        "pos": pos,
        "size": [520, 700],
        "flags": {},
        "order": 98,
        "mode": 0,
        "inputs": [
            {"name": "start_image", "type": "IMAGE", "link": None},
            {"name": "width", "type": "INT", "link": None, "widget": {"name": "width"}},
            {"name": "height", "type": "INT", "link": None, "widget": {"name": "height"}},
            {"name": "duration", "type": "FLOAT", "link": None, "widget": {"name": "duration"}},
        ],
        "outputs": [
            {"name": "visual_analysis", "type": "STRING", "links": []},
            {"name": "h3_prompt", "type": "STRING", "links": []},
            {"name": "length", "type": "INT", "links": []},
        ],
        "title": "I2V TWO-STAGE AUTO PROMPT",
        "properties": {"Node name for S&R": "MiniMaxH3I2VTwoStagePrompt"},
        "widgets_values": [
            "auto", VISUAL_MODEL, WRITER_MODEL,
            "Describe the subject moving naturally while preserving the starting image.",
            I2V_MANUAL, "", 1280, 720, 5.0, 180,
        ],
    }


def transform_i2v(name: str, integrated: bool) -> None:
    graph = load(name)
    if any(item["type"] == "MiniMaxH3I2VTwoStagePrompt" for item in graph["nodes"]):
        return
    old_ids = {
        item["id"] for item in graph["nodes"]
        if item["type"] in {
            "2af4e026-3781-4e28-963e-abf80e9930c3",
            "MiniMaxOpenRouterKey",
            "MiniMaxH3PromptMode",
            "PrimitiveStringMultiline",
        }
    }
    remove_nodes(graph, old_ids)
    if graph.get("definitions", {}).get("subgraphs"):
        graph["definitions"]["subgraphs"] = [
            subgraph for subgraph in graph["definitions"]["subgraphs"]
            if subgraph.get("id") != "2af4e026-3781-4e28-963e-abf80e9930c3"
        ]

    prompt_id = max(graph.get("last_node_id", 0), max(item["id"] for item in graph["nodes"])) + 1
    analysis_id = prompt_id + 1
    graph["last_node_id"] = analysis_id
    graph["nodes"].append(i2v_prompt_node(prompt_id, [-2320, 4520]))
    graph["nodes"].append(display_node(analysis_id, "Visual Analysis (Mistral)", [-1740, 4520]))

    add_link(graph, 175, 0, prompt_id, 0, "IMAGE")
    add_link(graph, 115, 0, prompt_id, 1, "INT")
    add_link(graph, 115, 1, prompt_id, 2, "INT")
    add_link(graph, 132, 0, prompt_id, 3, "FLOAT")
    add_link(graph, prompt_id, 0, analysis_id, 0, "STRING")
    add_link(graph, prompt_id, 1, 160, 0, "STRING")

    if integrated:
        add_link(graph, prompt_id, 1, 130, 4, "STRING")
        add_link(graph, prompt_id, 2, 130, 7, "INT")
        add_link(graph, prompt_id, 2, 141, 2, "INT")
        note = next((item for item in graph["nodes"] if item.get("title") == "Note: OpenRouter API key"), None)
        if note:
            note["widgets_values"] = [
                "## Two-stage Auto Prompt\n\nAUTO sends the start image to Mistral Medium 3.5, then sends only its text analysis to DeepSeek V4 Flash/Pro. MANUAL uses manual_prompt directly and makes zero OpenRouter calls. Set OPENROUTER_API_KEY or use the runtime key field."
            ]
    else:
        length_display = next(
            item for item in graph["nodes"] if item.get("title") == "H3 Frame Length"
        )
        add_link(graph, prompt_id, 2, length_display["id"], 0, "INT")

    for group in graph.get("groups", []):
        if group.get("title", "").startswith("I2V AUTO PROMPT"):
            group["title"] = "I2V TWO-STAGE AUTO PROMPT + MANUAL"
            group["bounding"] = [-2700, 4430, 1680, 1180]
    normalize_links(graph)
    save(name, graph)


def r2v_prompt_node(node_id: int, pos: list[float]) -> dict:
    inputs = [
        {"name": "width", "type": "INT", "link": None, "widget": {"name": "width"}},
        {"name": "height", "type": "INT", "link": None, "widget": {"name": "height"}},
        {"name": "duration", "type": "FLOAT", "link": None, "widget": {"name": "duration"}},
    ]
    inputs.extend(
        {"name": f"image_{index}", "type": "IMAGE", "link": None}
        for index in range(1, 10)
    )
    inputs.extend(
        {"name": f"video_{index}", "type": "IMAGE", "link": None}
        for index in range(1, 4)
    )
    inputs.extend(
        {"name": f"video_audio_{index}", "type": "AUDIO", "link": None}
        for index in range(1, 4)
    )
    inputs.extend(
        {"name": f"audio_{index}", "type": "AUDIO", "link": None}
        for index in range(1, 4)
    )
    return {
        "id": node_id,
        "type": "MiniMaxH3R2VTwoStagePrompt",
        "pos": pos,
        "size": [560, 780],
        "flags": {},
        "order": 99,
        "mode": 0,
        "inputs": inputs,
        "outputs": [
            {"name": "visual_analysis", "type": "STRING", "links": []},
            {"name": "h3_prompt", "type": "STRING", "links": []},
            {"name": "debug", "type": "STRING", "links": []},
        ],
        "title": "R2V TWO-STAGE AUTO PROMPT",
        "properties": {"Node name for S&R": "MiniMaxH3R2VTwoStagePrompt"},
        "widgets_values": [
            "auto", VISUAL_MODEL, WRITER_MODEL,
            "Create a coherent scene using the supplied references while preserving their defining traits.",
            R2V_MANUAL, 16, "", 1280, 720, 8.0, "auto", 180,
        ],
    }


def transform_r2v(name: str) -> None:
    graph = load(name)
    if any(item["type"] == "MiniMaxH3R2VTwoStagePrompt" for item in graph["nodes"]):
        return
    refpack = node(graph, 185)
    refpack["widgets_values"][0] = "References prepared for the two-stage Auto Prompt node."
    refpack["widgets_values"][3] = "none"
    refpack["widgets_values"][4] = ""
    refpack["widgets_values"][5] = ""
    refpack["widgets_values"][6] = "none"

    remove_links(graph, lambda link: link[1] == 185 and link[2] in (18, 19))
    prompt_id = max(graph.get("last_node_id", 0), max(item["id"] for item in graph["nodes"])) + 1
    analysis_id = prompt_id + 1
    debug_id = prompt_id + 2
    graph["last_node_id"] = debug_id
    graph["nodes"].append(r2v_prompt_node(prompt_id, [-1970, 4630]))
    graph["nodes"].append(display_node(analysis_id, "Visual Analysis (Mistral)", [-1360, 4590]))
    graph["nodes"].append(display_node(debug_id, "Auto Prompt Debug", [-1360, 4780]))

    add_link(graph, 115, 0, prompt_id, 0, "INT")
    add_link(graph, 115, 1, prompt_id, 1, "INT")
    add_link(graph, 132, 0, prompt_id, 2, "FLOAT")
    for output_slot in range(18):
        data_type = "IMAGE" if output_slot < 12 else "AUDIO"
        add_link(graph, 185, output_slot, prompt_id, output_slot + 3, data_type)
    add_link(graph, prompt_id, 0, analysis_id, 0, "STRING")
    add_link(graph, prompt_id, 1, 186, 0, "STRING")
    add_link(graph, prompt_id, 1, 184, 21, "STRING")
    add_link(graph, prompt_id, 2, debug_id, 0, "STRING")

    note = next((item for item in graph["nodes"] if item.get("title") == "Note: HearmemanAI"), None)
    if note:
        note["widgets_values"] = [
            "## MiniMax H3 R2V Two-Stage Auto Prompt\n\nMiniMaxRefPack remains the reference loader and sends all image, video, and audio sockets to MiniMax. It makes no prompt API call. AUTO analyzes standalone images plus 16 evenly sampled frames per cropped/trimmed video with Mistral, then sends text only to DeepSeek. Audio is preserved for MiniMax but is not analyzed. MANUAL makes zero OpenRouter calls."
        ]
    for group in graph.get("groups", []):
        if group.get("title", "").startswith("R2V AUTO PROMPT"):
            group["title"] = "R2V TWO-STAGE AUTO PROMPT + MANUAL"
            group["bounding"] = [-2640, 4550, 2580, 1530]
    normalize_links(graph)
    save(name, graph)


def main() -> None:
    transform_i2v("I2V_AutoPrompt_Module.json", integrated=False)
    transform_i2v("I2V_AutoPrompt_Integrated.json", integrated=True)
    transform_r2v("R2V_AutoPrompt_Module.json")
    transform_r2v("R2V_AutoPrompt_Integrated.json")


if __name__ == "__main__":
    main()
