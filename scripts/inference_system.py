"""
Inference System for Fine-tuned Wheel Loader Control Model
Runs inference on new images using the fine-tuned model + object detection
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel


class ObjectDetector:
    """Detect objects in images for context."""

    def __init__(self, enable: bool = True):
        """
        Initialize object detector.

        Args:
            enable: Enable object detection
        """
        self.enabled = enable
        self.detector = None

        if enable:
            try:
                print("Loading object detector...")
                self.detector = pipeline(
                    "zero-shot-object-detection",
                    model="google/owlv2-base-patch16-ensemble",
                    device=0  # 0GPU -1CPU
                )
                self.target_classes = [
                    "pile of gravel", "pile of dirt", "pile of sand", "material pile",
                    "excavation area", "dump truck", "loader bucket", "construction vehicle"
                ]
                print("✓ Object detector loaded")
            except Exception as e:
                print(f"Warning: Could not load detector: {e}")
                self.detector = None
                self.enabled = False

    def detect(self, image_path: str) -> List[Dict]:
        """Detect objects in image."""
        if not self.enabled or not self.detector:
            return []

        try:
            image = Image.open(image_path)
            width, height = image.size

            predictions = self.detector(
                image,
                candidate_labels=self.target_classes,
                threshold=0.15
            )

            objects = []
            for pred in predictions:
                bbox = pred['box']
                x_min, y_min = bbox['xmin'] / width, bbox['ymin'] / height
                x_max, y_max = bbox['xmax'] / width, bbox['ymax'] / height
                center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2

                h_pos = "left" if center_x < 0.33 else "center" if center_x < 0.67 else "right"
                v_pos = "top" if center_y < 0.33 else "middle" if center_y < 0.67 else "bottom"

                objects.append({
                    "label": pred['label'],
                    "confidence": float(pred['score']),
                    "position": f"{v_pos}-{h_pos}",
                    "center": [float(center_x), float(center_y)]
                })

            return objects
        except Exception as e:
            print(f"Detection error: {e}")
            return []


class LoaderInferenceSystem:
    """Inference system for wheel loader control."""

    def __init__(self,
                 model_path: str,
                 base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 enable_detection: bool = True,
                 device: str = "cpu"):
        """
        Initialize inference system.

        Args:
            model_path: Path to fine-tuned model (LoRA weights)
            base_model: Base model used for training
            enable_detection: Enable object detection
            device: Device for inference ("cpu" or "cuda")
        """
        self.model_path = model_path
        self.base_model = base_model
        self.device = device

        print(f"Loading model from {model_path}...")
        print(f"Base model: {base_model}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        print("Loading base model...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )

        # Load LoRA weights
        print("Loading LoRA weights...")
        self.model = PeftModel.from_pretrained(base, model_path)
        self.model.eval()

        if device == "cpu":
            self.model = self.model.to(device)

        print("✓ Model loaded successfully")

        # Initialize detector
        self.detector = ObjectDetector(enable=enable_detection)

    def format_prompt(self, question: str, detected_objects: List[Dict] = None) -> str:
        """
        Format prompt for the model.

        Args:
            question: User question
            detected_objects: Optional detected objects for context

        Returns:
            Formatted prompt
        """
        system_prompt = (
            "You are an AI assistant for wheel loader operations at construction sites. "
            "Provide specific, actionable answers based on visual information about the scene. "
            "When answering questions about positions or locations, use precise coordinates and descriptions."
        )

        context = ""
        if detected_objects:
            obj_desc = ", ".join([
                f"{obj['label']} at {obj['position']}"
                for obj in detected_objects[:3]
            ])
            context = f"Objects in scene: {obj_desc}\n\n"

        prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{context}{question} [/INST]"""

        return prompt

    def generate_answer(self,
                        question: str,
                        image_path: Optional[str] = None,
                        max_new_tokens: int = 150,
                        temperature: float = 0.7,
                        top_p: float = 0.9) -> Dict:
        """
        Generate answer to a question.

        Args:
            question: Question to answer
            image_path: Optional image path for context
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            Dictionary with answer and metadata
        """
        # Detect objects if image provided
        detected_objects = []
        if image_path and Path(image_path).exists():
            detected_objects = self.detector.detect(image_path)

        # Format prompt
        prompt = self.format_prompt(question, detected_objects)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the answer (after [/INST])
        if "[/INST]" in full_response:
            answer = full_response.split("[/INST]")[-1].strip()
        else:
            answer = full_response.strip()

        return {
            "question": question,
            "answer": answer,
            "image": image_path,
            "detected_objects": detected_objects,
            "num_objects": len(detected_objects)
        }

    def interactive_session(self):
        """Start interactive Q&A session."""
        print("\n" + "=" * 70)
        print("WHEEL LOADER CONTROL - INTERACTIVE INFERENCE")
        print("=" * 70)
        print("Commands:")
        print("  'q <question>' - Ask a question")
        print("  'image <path>' - Set current image context")
        print("  'clear' - Clear image context")
        print("  'exit' - Exit session")
        print("=" * 70 + "\n")

        current_image = None

        while True:
            try:
                user_input = input("\n> ").strip()

                if not user_input:
                    continue

                if user_input.lower() == "exit":
                    print("Goodbye!")
                    break

                if user_input.lower() == "clear":
                    current_image = None
                    print("Image context cleared")
                    continue

                if user_input.startswith("image "):
                    img_path = user_input[6:].strip()
                    if Path(img_path).exists():
                        current_image = img_path
                        print(f"✓ Image context set: {img_path}")

                        # Show detected objects
                        if self.detector.enabled:
                            objects = self.detector.detect(img_path)
                            if objects:
                                print(f"  Detected {len(objects)} objects:")
                                for obj in objects[:5]:
                                    print(f"    - {obj['label']} at {obj['position']}")
                    else:
                        print(f"✗ Image not found: {img_path}")
                    continue

                if user_input.startswith("q "):
                    question = user_input[2:].strip()

                    if not question:
                        print("Please provide a question")
                        continue

                    print("\nThinking...")
                    result = self.generate_answer(question, current_image)

                    print(f"\nQ: {result['question']}")
                    print(f"A: {result['answer']}")

                    if result['detected_objects']:
                        print(f"\n(Context: {result['num_objects']} objects detected)")
                else:
                    print("Unknown command. Use 'q <question>', 'image <path>', or 'exit'")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

    def batch_inference(self,
                        questions: List[str],
                        image_path: Optional[str] = None,
                        output_file: Optional[str] = None):
        """
        Run inference on multiple questions.

        Args:
            questions: List of questions
            image_path: Optional image context
            output_file: Optional output JSON file
        """
        results = []

        print(f"\nRunning batch inference on {len(questions)} questions...")

        for i, question in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {question}")

            result = self.generate_answer(question, image_path)
            results.append(result)

            print(f"  → {result['answer'][:100]}...")

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"\n✓ Results saved to {output_file}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with fine-tuned wheel loader model"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned model directory"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model used for training"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for inference"
    )

    # Inference mode
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive session"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Single question to answer"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Image path for context"
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        help="JSON file with list of questions"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for batch results"
    )

    # Options
    parser.add_argument(
        "--no-detection",
        action="store_true",
        help="Disable object detection"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Maximum tokens to generate (default: 150)"
    )

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model directory not found: {args.model}")
        print("\nMake sure you've trained a model first:")
        print("  python train_model.py --vqa-file <file> --output-dir <dir>")
        return

    print("=" * 70)
    print("WHEEL LOADER INFERENCE SYSTEM")
    print("=" * 70)

    # Initialize system
    system = LoaderInferenceSystem(
        model_path=args.model,
        base_model=args.base_model,
        enable_detection=not args.no_detection,
        device=args.device
    )

    # Run inference
    if args.interactive:
        system.interactive_session()

    elif args.question:
        result = system.generate_answer(
            question=args.question,
            image_path=args.image,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens
        )

        print("\n" + "=" * 70)
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}")

        if result['detected_objects']:
            print(f"\nDetected objects:")
            for obj in result['detected_objects']:
                print(f"  - {obj['label']} at {obj['position']} "
                      f"(confidence: {obj['confidence']:.2f})")
        print("=" * 70)

    elif args.questions_file:
        with open(args.questions_file, 'r') as f:
            questions = json.load(f)

        if not isinstance(questions, list):
            print("Error: Questions file must contain a JSON array")
            return

        system.batch_inference(
            questions=questions,
            image_path=args.image,
            output_file=args.output
        )

    else:
        print("\nNo inference mode specified. Use one of:")
        print("  --interactive           Start interactive session")
        print("  --question <text>       Answer single question")
        print("  --questions-file <file> Batch inference")
        print("\nExample:")
        print(f"  python inference_system.py --model {args.model} --interactive")


if __name__ == "__main__":
    main()
