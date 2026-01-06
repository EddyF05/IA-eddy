import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

torch.set_num_threads(4)
torch.set_num_interop_threads(4)

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "models", "tutor-lora"))


def load_model():
    print("⏳ Cargando modelo base...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=False
    )

    model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
    )

    print("⏳ Cargando adaptador LoRA...")
    model = PeftModel.from_pretrained(
        model,
        LORA_PATH,
        device_map=None
    )

    model.eval()
    print("✅ Modelo listo\n")
    return tokenizer, model

def infer(prompt, tokenizer, model):
    messages = [
        {"role": "system", "content": "Eres un tutor educativo experto. Explicas de forma clara, precisa y pedagógica."},
        {"role": "user", "content": prompt}
    ]
    
    template = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    inputs = tokenizer(template, return_tensors="pt").to(model.device)
    
    stop_tokens = ["<|end|>", "<|user|>", "<|system|>"]
    stop_token_ids = [tokenizer.encode(token, add_special_tokens=False)[0] for token in stop_tokens if token in tokenizer.get_vocab()]

    with torch.no_grad():
       outputs = model.generate(
    **inputs,
    max_new_tokens=64,    
    do_sample=False,        
    temperature=0.0,
    top_p=1.0,
    num_beams=1
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    if "<|assistant|>" in result:
        result = result.split("<|assistant|>")[-1]
    
    for stop_token in stop_tokens:
        if stop_token in result:
            result = result.split(stop_token)[0]
    
    return result.strip()

def main():
    print("\n" + "="*50)
    print("   🎓  Tutor IA - Asistente Educativo")
    print("="*50)
    
    tokenizer, model = load_model()
    
    print("¡Hola! Soy tu tutor de IA. Pregúntame lo que quieras aprender.")
    print("Escribe 'salir' para terminar la conversación.\n")

    while True:
        user_input = input("💬 Tú: ")

        if user_input.lower() in ["salir", "exit", "quit"]:
            print("\n¡Hasta luego! Sigue aprendiendo 👋\n")
            break

        respuesta = infer(user_input, tokenizer, model)
        print(f"\n🤖 Tutor: {respuesta}\n")


if __name__ == "__main__":
    main()
