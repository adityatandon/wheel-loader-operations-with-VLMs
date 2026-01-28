"""
VQA Generator with Object Detection
Combines Moondream2 for VQA + OWLv2 for object detection
Optimized for CPU execution
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import argparse

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, pipeline
from tqdm import tqdm

# CPU optimization
#torch.set_num_threads(4)
#torch.set_num_interop_threads(1)


class VQAGenerator:
    """Generate VQA pairs using Moondream2."""

    def __init__(self,
                 model_id: str = "vikhyatk/moondream2",
                 max_edge: int = 768,
                 num_questions: int = 5):
        """
        Initialize VQA generator.

        Args:
            model_id: Moondream2 model ID
            max_edge: Max image dimension (smaller = faster)
            num_questions: Questions to generate per image
        """
        print("Loading Moondream2 model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            revision="2025-01-09",
            #device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
            torch_dtype=torch.float32,
            force_download=True,
        ).to("cuda")

        # Patch
        if not hasattr(self.model, "all_tied_weights_keys"):
            self.model.all_tied_weights_keys = getattr(self.model, "_tied_weights_keys", {})

        print("✓ Moondream2 loaded")

        self.max_edge = max_edge
        self.num_questions = num_questions
        self.settings = {"temperature": 0.3, "top_p": 0.9, "max_tokens": 128}

    def load_and_resize(self, path: str) -> Image.Image:
        """Load and resize image."""
        img = Image.open(path).convert("RGB")
        w, h = img.size
        m = max(w, h)
        if m > self.max_edge:
            scale = self.max_edge / m
            img = img.resize((int(w * scale), int(h * scale)))
        return img

    def parse_questions(self, text: str, k: int) -> List[str]:
        """Parse questions from model output."""
        text = text.strip()

        # Try JSON first
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()][:k]
        except:
            pass

        # Fallback: parse lines
        qs = []
        for line in text.splitlines():
            line = line.strip().lstrip("-•*").strip()
            if not line:
                continue
            if line[0].isdigit():
                line = line.lstrip("0123456789").lstrip(").").strip()
            if line:
                qs.append(line)
        return qs[:k]

    def generate_vqa(self, image_path: str) -> List[Dict]:
        """
        Generate VQA pairs for an image.

        Args:
            image_path: Path to image

        Returns:
            List of Q&A pairs
        """
        img = self.load_and_resize(image_path)

        # Encode image once
        enc = self.model.encode_image(img)

        # Generate questions
        q_prompt = (
            f"Generate {self.num_questions} diverse visual questions about this construction site image. "
            f"Focus on the type of objects, the positions of the objects, loader operations, material piles, digging and loading sequences. "
            "Return ONLY a JSON array of strings."
        )

        q_resp = self.model.query(enc, q_prompt, settings=self.settings)
        q_text = q_resp["answer"] if isinstance(q_resp, dict) else str(q_resp)
        questions = self.parse_questions(q_text, self.num_questions)

        if not questions:
            print(f"  Warning: No questions parsed for {Path(image_path).name}")
            return []

        # Answer questions
        vqa_pairs = []
        for q in questions:
            a_resp = self.model.query(
                enc,
                f"Answer concisely (short phrase or one sentence). Question: {q}",
                settings=self.settings
            )
            ans = a_resp["answer"] if isinstance(a_resp, dict) else str(a_resp)

            vqa_pairs.append({
                "question": q,
                "answer": ans.strip(),
                "type": "moondream_generated"
            })

        return vqa_pairs


class ObjectDetector:
    """Detect objects using OWLv2 zero-shot detection."""

    def __init__(self):
        """Initialize object detector."""
        print("Loading OWLv2 object detector...")
        try:
            self.detector = pipeline(
                "zero-shot-object-detection",
                model="google/owlv2-base-patch16-ensemble",
                device=0  # CPU
            )
            print("✓ OWLv2 detector loaded")
        except Exception as e:
            print(f"✗ Could not load detector: {e}")
            self.detector = None

        self.target_classes = [
            "pile of gravel", "pile of dirt", "pile of sand", "pile of rocks",
            "material pile", "excavation area", "dump truck", "crusher", "bulldozer",
            "construction vehicle", "loader bucket", "wheel loader", "fuel tank", "people"
        ]

    def detect(self, image_path: str, threshold: float = 0.15) -> List[Dict]:
        """
        Detect objects in image.

        Args:
            image_path: Path to image
            threshold: Confidence threshold

        Returns:
            List of detected objects with coordinates
        """
        if not self.detector:
            return []

        try:
            image = Image.open(image_path)
            width, height = image.size

            predictions = self.detector(
                image,
                candidate_labels=self.target_classes,
                threshold=threshold
            )

            objects = []
            for pred in predictions:
                bbox = pred['box']

                # Normalized coordinates (0-1)
                x_min = bbox['xmin'] / width
                y_min = bbox['ymin'] / height
                x_max = bbox['xmax'] / width
                y_max = bbox['ymax'] / height

                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2

                # Position description
                h_pos = "left" if center_x < 0.33 else "center" if center_x < 0.67 else "right"
                v_pos = "top" if center_y < 0.33 else "middle" if center_y < 0.67 else "bottom"

                objects.append({
                    "label": pred['label'],
                    "confidence": float(pred['score']),
                    "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)],
                    "center": [float(center_x), float(center_y)],
                    "position": f"{v_pos}-{h_pos}",
                    "pixel_bbox": {
                        "xmin": int(bbox['xmin']),
                        "ymin": int(bbox['ymin']),
                        "xmax": int(bbox['xmax']),
                        "ymax": int(bbox['ymax'])
                    }
                })

            return objects

        except Exception as e:
            print(f"  Detection error: {e}")
            return []


class ActionPlanner:
    """Generate actionable commands from detections."""

    @staticmethod
    def generate_action_plan(detected_objects: List[Dict], task: str = "fill the bucket") -> List[Dict]:
        """
        Generate action sequence for a task.

        Args:
            detected_objects: List of detected objects
            task: Task to perform

        Returns:
            List of action steps
        """
        if not detected_objects:
            return []

        piles = [obj for obj in detected_objects if "pile" in obj["label"].lower()]

        if not piles:
            return []

        # Find nearest pile (closest vertically, assuming loader at bottom)
        nearest = min(piles, key=lambda o: o["center"][1])

        if "fill" in task.lower() or "load" in task.lower() or "scoop" in task.lower():
            return [
                {
                    "action": "navigate",
                    "description": f"Drive to {nearest['label']} at {nearest['position']}",
                    "target": nearest['label'],
                    "target_position": nearest['center']
                },
                {
                    "action": "lower_bucket",
                    "description": "Lower bucket to ground level"
                },
                {
                    "action": "tilt_forward",
                    "description": "Tilt bucket forward 15 degrees"
                },
                {
                    "action": "scoop",
                    "description": f"Scoop material from {nearest['label']}"
                },
                {
                    "action": "raise_bucket",
                    "description": "Raise bucket to transport height"
                }
            ]

        return []


class IntegratedVQASystem:
    """Complete VQA + Detection + Action Planning system."""

    def __init__(self,
                 num_questions: int = 5,
                 enable_detection: bool = True,
                 enable_actions: bool = True):
        """
        Initialize system.

        Args:
            num_questions: Questions per image
            enable_detection: Enable object detection
            enable_actions: Enable action planning
        """
        self.vqa_gen = VQAGenerator(num_questions=num_questions)
        self.detector = ObjectDetector() if enable_detection else None
        self.enable_actions = enable_actions

    def process_image(self, image_path: str) -> Dict:
        """
        Process single image: VQA + Detection + Actions.

        Args:
            image_path: Path to image

        Returns:
            Complete results dictionary
        """
        # Generate VQA
        vqa_pairs = self.vqa_gen.generate_vqa(image_path)

        # Detect objects
        detected_objects = []
        if self.detector:
            detected_objects = self.detector.detect(image_path)

        # Add spatial questions
        spatial_vqa = []
        if detected_objects:
            # Question: What objects are detected?
            obj_labels = [obj["label"] for obj in detected_objects[:3]]
            spatial_vqa.append({
                "question": "What objects are detected in the scene?",
                "answer": f"Detected: {', '.join(obj_labels)}",
                "type": "object_detection"
            })

            # Question: Where is nearest pile?
            piles = [obj for obj in detected_objects if "pile" in obj["label"].lower()]
            if piles:
                nearest = min(piles, key=lambda o: o["center"][1])
                spatial_vqa.append({
                    "question": "Where is the nearest material pile?",
                    "answer": f"The {nearest['label']} is at {nearest['position']} "
                              f"(center coordinates: {nearest['center'][0]:.2f}, {nearest['center'][1]:.2f})",
                    "type": "spatial_detection"
                })

        # Generate action plans
        action_plans = []
        if self.enable_actions and detected_objects:
            actions = ActionPlanner.generate_action_plan(detected_objects)
            if actions:
                action_plans.append({
                    "question": "What should the loader do to fill the bucket?",
                    "answer": " → ".join([a["description"] for a in actions]),
                    "type": "action_planning",
                    "action_sequence": actions
                })

        return {
            "image": str(image_path),
            "vqa_pairs": vqa_pairs,
            "spatial_vqa": spatial_vqa,
            "action_plans": action_plans,
            "detected_objects": detected_objects,
            "total_qa_pairs": len(vqa_pairs) + len(spatial_vqa) + len(action_plans)
        }

    def process_dataset(self,
                        image_paths: List[str],
                        output_jsonl: str,
                        output_summary: Optional[str] = None):
        """
        Process multiple images and save results.

        Args:
            image_paths: List of image paths
            output_jsonl: Output JSONL file (one Q&A per line)
            output_summary: Optional summary JSON file
        """
        output_path = Path(output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_results = []
        total_qa_pairs = 0

        with output_path.open("w", encoding="utf-8") as f:
            for i, img_path in enumerate(tqdm(image_paths, desc="Processing images"), 1):
                print(f"\n[{i}/{len(image_paths)}] Processing {Path(img_path).name}")

                try:
                    result = self.process_image(img_path)

                    # Write VQA pairs as JSONL
                    for vqa in result["vqa_pairs"]:
                        row = {
                            "image": result["image"],
                            "question": vqa["question"],
                            "answer": vqa["answer"],
                            "type": vqa["type"]
                        }
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

                    # Write spatial Q&A
                    for vqa in result["spatial_vqa"]:
                        row = {
                            "image": result["image"],
                            "question": vqa["question"],
                            "answer": vqa["answer"],
                            "type": vqa["type"]
                        }
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

                    # Write action plans
                    for vqa in result["action_plans"]:
                        row = {
                            "image": result["image"],
                            "question": vqa["question"],
                            "answer": vqa["answer"],
                            "type": vqa["type"],
                            "action_sequence": vqa.get("action_sequence", [])
                        }
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

                    all_results.append(result)
                    total_qa_pairs += result["total_qa_pairs"]

                    print(f"  ✓ Generated {result['total_qa_pairs']} Q&A pairs")
                    print(f"  ✓ Detected {len(result['detected_objects'])} objects")

                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    continue

        # Save summary
        if output_summary:
            summary_path = Path(output_summary)
            summary_path.parent.mkdir(parents=True, exist_ok=True)

            with summary_path.open("w", encoding="utf-8") as f:
                json.dump({
                    "total_images": len(all_results),
                    "total_qa_pairs": total_qa_pairs,
                    "avg_qa_per_image": total_qa_pairs / len(all_results) if all_results else 0,
                    "results": all_results
                }, f, indent=2)

            print(f"\n✓ Summary saved to {output_summary}")

        print(f"\n{'=' * 70}")
        print("GENERATION COMPLETE")
        print(f"{'=' * 70}")
        print(f"Total images: {len(all_results)}")
        print(f"Total Q&A pairs: {total_qa_pairs}")
        print(f"Average Q&A per image: {total_qa_pairs / len(all_results):.1f}")
        print(f"Output JSONL: {output_jsonl}")
        print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate VQA + Object Detection for wheel loader images"
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        help="Directory containing frames (searches recursively for images)"
    )
    parser.add_argument(
        "--images",
        nargs="+",
        help="Specific image paths (alternative to --frames-dir)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vqa_dataset.jsonl",
        help="Output JSONL file"
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="vqa_summary.json",
        help="Output summary JSON file"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=5,
        help="Number of questions per image (default: 5)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of images to process (for testing)"
    )
    parser.add_argument(
        "--no-detection",
        action="store_true",
        help="Disable object detection (faster, Moondream only)"
    )
    parser.add_argument(
        "--no-actions",
        action="store_true",
        help="Disable action planning"
    )

    args = parser.parse_args()

    # Get image paths
    image_paths = []

    if args.images:
        image_paths = [Path(p) for p in args.images]
    elif args.frames_dir:
        frames_dir = Path(args.frames_dir)
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_paths.extend(frames_dir.rglob(ext))
    else:
        print("Error: Must specify either --frames-dir or --images")
        return

    if args.limit:
        image_paths = image_paths[:args.limit]

    if not image_paths:
        print("Error: No images found")
        return

    print("=" * 70)
    print("VQA GENERATOR: Moondream2 + OWLv2 Object Detection")
    print("=" * 70)
    print(f"Images to process: {len(image_paths)}")
    print(f"Questions per image: {args.num_questions}")
    print(f"Object detection: {'Disabled' if args.no_detection else 'Enabled'}")
    print(f"Action planning: {'Disabled' if args.no_actions else 'Enabled'}")
    print(f"Output: {args.output}")
    print("=" * 70)

    # Initialize system
    system = IntegratedVQASystem(
        num_questions=args.num_questions,
        enable_detection=not args.no_detection,
        enable_actions=not args.no_actions
    )

    # Process images
    system.process_dataset(
        image_paths=[str(p) for p in image_paths],
        output_jsonl=args.output,
        output_summary=args.summary
    )


if __name__ == "__main__":
    main()
