"""
LLM Fine-tuning Script for Wheel Loader Control
Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning on VQA data
specific to wheel loader operations at construction sites.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import argparse

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class VQADataProcessor:
    """Process VQA JSONL into training format."""

    def __init__(self, vqa_jsonl: str):
        """
        Load VQA dataset.

        Args:
            vqa_jsonl: Path to VQA JSONL file
        """
        print(f"Loading VQA data from {vqa_jsonl}...")
        self.vqa_data = []

        with open(vqa_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    self.vqa_data.append(item)
                except json.JSONDecodeError:
                    continue

        print(f"✓ Loaded {len(self.vqa_data)} Q&A pairs")

    def format_for_training(self, system_prompt: Optional[str] = None) -> List[str]:
        """
        Format VQA data for training.

        Args:
            system_prompt: Optional system prompt

        Returns:
            List of formatted training texts
        """
        if system_prompt is None:
            system_prompt = (
                "You are an AI assistant for wheel loader operations at construction sites. "
                "Provide specific, actionable answers based on visual information about the scene. "
                "When answering questions about positions or locations, use precise coordinates and descriptions."
            )

        formatted_texts = []

        for item in self.vqa_data:
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()

            if not question or not answer:
                continue

            # Add context if available
            context = ""
            if "detected_objects" in item and item["detected_objects"]:
                objects = item["detected_objects"]
                obj_desc = ", ".join([
                    f"{obj['label']} at {obj['position']}"
                    for obj in objects[:3]
                ])
                context = f"Objects in scene: {obj_desc}\n\n"

            # Format as Llama-2 chat template
            formatted_text = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{context}{question} [/INST] {answer}</s>"""

            formatted_texts.append(formatted_text)

        return formatted_texts

    def create_dataset(self, system_prompt: Optional[str] = None) -> Dataset:
        """
        Create Hugging Face Dataset.

        Args:
            system_prompt: Optional system prompt

        Returns:
            Hugging Face Dataset
        """
        texts = self.format_for_training(system_prompt)

        return Dataset.from_dict({"text": texts})


class LoRATrainer:
    """Fine-tune LLM using LoRA."""

    def __init__(self,
                 base_model: str = "meta-llama/Llama-2-7b-hf",
                 use_8bit: bool = True,
                 use_4bit: bool = False):
        """
        Initialize trainer.

        Args:
            base_model: Base model name/path
            use_8bit: Use 8-bit quantization
            use_4bit: Use 4-bit quantization (more aggressive)
        """
        self.base_model = base_model
        self.use_8bit = use_8bit
        self.use_4bit = use_4bit

        print(f"Base model: {base_model}")
        print(f"Quantization: {'4-bit' if use_4bit else '8-bit' if use_8bit else 'None'}")

    def load_model_and_tokenizer(self):
        """Load base model and tokenizer with quantization."""
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        print("Loading model...")

        # Quantization config
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.use_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            bnb_config = None

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16 if (self.use_4bit or self.use_8bit) else torch.float32,
            trust_remote_code=True,
        )

        # Prepare for training
        if self.use_8bit or self.use_4bit:
            model = prepare_model_for_kbit_training(model)

        print("✓ Model and tokenizer loaded")

        return model, tokenizer

    def setup_lora(self, model, lora_r: int = 16, lora_alpha: int = 32):
        """
        Setup LoRA configuration.

        Args:
            model: Base model
            lora_r: LoRA rank (lower = fewer parameters)
            lora_alpha: LoRA alpha (scaling factor)

        Returns:
            Model with LoRA
        """
        print("\nConfiguring LoRA...")

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        print(f"✓ LoRA configured")
        print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print(f"  Total parameters: {total_params:,}")

        return model

    def train(self,
              model,
              tokenizer,
              dataset: Dataset,
              output_dir: str,
              num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 2e-4,
              max_seq_length: int = 512,
              gradient_accumulation_steps: int = 4,
              save_steps: int = 100,
              logging_steps: int = 10):
        """
        Train the model.

        Args:
            model: Model with LoRA
            tokenizer: Tokenizer
            dataset: Training dataset
            output_dir: Output directory
            num_epochs: Number of training epochs
            batch_size: Batch size per device
            learning_rate: Learning rate
            max_seq_length: Maximum sequence length
            gradient_accumulation_steps: Gradient accumulation steps
            save_steps: Save checkpoint every N steps
            logging_steps: Log every N steps
        """
        print("\nPreparing dataset...")

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_seq_length,
                padding="max_length"
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        print(f"✓ Dataset tokenized: {len(tokenized_dataset)} examples")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=50,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=3,
            fp16=True,
            optim="paged_adamw_8bit" if (self.use_8bit or self.use_4bit) else "adamw_torch",
            report_to="none",  # Disable wandb/tensorboard
            remove_unused_columns=False,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Training examples: {len(tokenized_dataset)}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Gradient accumulation: {gradient_accumulation_steps}")
        print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
        print(f"Learning rate: {learning_rate}")
        print(f"Output directory: {output_dir}")
        print("=" * 70 + "\n")

        # Train
        trainer.train()

        # Save final model
        print("\nSaving final model...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"\n✓ Training complete! Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune LLM for wheel loader control"
    )

    # Data arguments
    parser.add_argument(
        "--vqa-file",
        type=str,
        required=True,
        help="Input VQA JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="finetuned_loader_model",
        help="Output directory for fine-tuned model"
    )

    # Model arguments
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Base model (default: Llama-2-7B, alternative: TinyLlama/TinyLlama-1.1B-Chat-v1.0)"
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization (most memory efficient)"
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Disable quantization (requires more memory)"
    )

    # LoRA arguments
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16, lower = fewer parameters)"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)"
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per device (default: 4)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("LLM FINE-TUNING FOR WHEEL LOADER CONTROL")
    print("=" * 70)

    # Check if VQA file exists
    if not Path(args.vqa_file).exists():
        print(f"Error: VQA file not found: {args.vqa_file}")
        return

    # Process VQA data
    processor = VQADataProcessor(args.vqa_file)
    dataset = processor.create_dataset()

    print(f"\nDataset statistics:")
    print(f"  Total Q&A pairs: {len(dataset)}")

    if len(dataset) < 50:
        print("\n⚠ WARNING: Dataset is very small (< 50 examples)")
        print("  The model will likely overfit. Consider generating more VQA pairs.")
        response = input("\nContinue anyway? (yes/no): ")
        if response.lower() != "yes":
            print("Training cancelled.")
            return

    # Initialize trainer
    trainer = LoRATrainer(
        base_model=args.base_model,
        use_8bit=not args.no_quantization and not args.use_4bit,
        use_4bit=args.use_4bit
    )

    # Load model
    model, tokenizer = trainer.load_model_and_tokenizer()

    # Setup LoRA
    model = trainer.setup_lora(model, lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    # Train
    trainer.train(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        gradient_accumulation_steps=args.gradient_accumulation,
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()