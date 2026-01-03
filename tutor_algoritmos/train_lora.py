from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

# -----------------------------
# 1. Cargar dataset
# -----------------------------
dataset = load_dataset(
    "json",
    data_files="tutor_programacion.jsonl",
    split="train"
)

# -----------------------------
# 2. Cargar modelo base
# -----------------------------
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name
)

# -----------------------------
# 3. Configurar LoRA
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# -----------------------------
# 4. Preprocesar dataset
# -----------------------------
def format_instruction(example):
    prompt = f"Instrucción: {example['instruction']}\nRespuesta:"
    return tokenizer(
        prompt + example["response"],
        truncation=True,
        max_length=512
    )

tokenized_dataset = dataset.map(format_instruction)

# -----------------------------
# 5. Argumentos de entrenamiento
# -----------------------------
training_args = TrainingArguments(
    output_dir="./lora-tutor",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=10,
    save_steps=500,
    report_to="none"
)

# -----------------------------
# 6. Entrenador
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    ),
)

# -----------------------------
# 7. Entrenamiento
# -----------------------------
trainer.train()

# -----------------------------
# 8. Guardar adaptadores
# -----------------------------
model.save_pretrained("./lora-tutor")
