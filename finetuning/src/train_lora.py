import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_PATH = "../models/tutor-lora"
DATA_DIR = "../data/processed"

CONFIG = {
    "max_seq_length": 128,
    "num_epochs": 3,
    "batch_size": 1,
    "gradient_accumulation": 8,
    "lr": 1e-4,
    "warmup_steps": 5,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05
}

def prepare_training_text(sample):
    formatted = (
        f"### Instrucción:\n{sample['instruction']}\n\n"
        f"### Respuesta:\n{sample['output']}</s>"
    )
    return {"text": formatted}

def load_and_prepare_data():
    data_files = {
        "train": f"{DATA_DIR}/train.jsonl",
        "val": f"{DATA_DIR}/val.jsonl"
    }
    dataset = load_dataset("json", data_files=data_files)
    return dataset.map(prepare_training_text)

def setup_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def tokenize_dataset(dataset, tokenizer):
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=CONFIG["max_seq_length"]
        )
    return dataset.map(tokenize_batch, batched=True, remove_columns=["text"])

def configure_lora_model(base_model):
    lora_config = LoraConfig(
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"]
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model

def main():
    print("⚙️  Preparando datos...")
    dataset = load_and_prepare_data()

    
    print("🔤 Configurando tokenizer...")
    tokenizer = setup_tokenizer()
    
    print("📊 Tokenizando dataset...")
    tokenized_data = tokenize_dataset(dataset, tokenizer)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"🚀 Cargando modelo base ({device})...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype)
    base_model.config.use_cache = False
    
    print("🔧 Aplicando adaptadores LoRA...")
    model = configure_lora_model(base_model)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation"],
        num_train_epochs=CONFIG["num_epochs"],
        learning_rate=CONFIG["lr"],
        warmup_steps=CONFIG["warmup_steps"],
        logging_steps=10,
        save_steps=500,
        fp16=False,
        dataloader_pin_memory=False,
        do_eval=False,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        data_collator=collator
    )
    
    print("\n🎯 Iniciando entrenamiento...\n")
    trainer.train()
    
    print("\n💾 Guardando modelo...")
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    
    print("✅ Entrenamiento completado.\n")

if __name__ == "__main__":
    main()
