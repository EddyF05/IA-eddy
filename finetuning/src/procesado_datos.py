import json
import random
from pathlib import Path

# ===============================
# RUTAS
# ===============================

RAW_DIR = Path("../datos/raw")
PROCESSED_DIR = Path("../datos/processed")

# Puedes cambiar entre los dos formatos:
# RAW_FILE = RAW_DIR / "ejemplos.md"          # Formato Markdown
RAW_FILE = RAW_DIR / "conversaciones_raw.txt"  # Formato ALUMNO/TUTOR

TRAIN_FILE = PROCESSED_DIR / "train.jsonl"
VAL_FILE = PROCESSED_DIR / "val.jsonl"
TEST_FILE = PROCESSED_DIR / "test.jsonl"

# ===============================
# PROPORCIONES
# ===============================

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def parse_markdown(md_text: str):
    """
    Extrae pares instruction / output desde el markdown
    """
    examples = []
    blocks = md_text.split("---")

    for block in blocks:
        if "### Instrucción" in block and "### Respuesta" in block:
            try:
                instruction = block.split("### Instrucción")[1].split("### Respuesta")[0].strip()
                output = block.split("### Respuesta")[1].strip()

                if instruction and output:
                    examples.append({
                        "instruction": instruction,
                        "input": "",
                        "output": output
                    })
            except IndexError:
                continue

    return examples


def parse_conversaciones(text: str):
    """
    Extrae pares instruction / output desde formato ALUMNO/TUTOR
    Formato: ALUMNO: pregunta | TUTOR: respuesta
    """
    examples = []
    lines = text.strip().split("\n")
    
    for line in lines:
        line = line.strip()
        # Saltar líneas vacías, comentarios o encabezados
        if not line or line.startswith("#"):
            continue
            
        # Buscar el patrón ALUMNO: ... | TUTOR: ...
        if "ALUMNO:" in line and "TUTOR:" in line:
            try:
                # Dividir por el separador |
                parts = line.split("|")
                if len(parts) >= 2:
                    # Extraer la pregunta (después de ALUMNO:)
                    instruction = parts[0].replace("ALUMNO:", "").strip()
                    # Extraer la respuesta (después de TUTOR:)
                    output = parts[1].replace("TUTOR:", "").strip()
                    
                    if instruction and output:
                        examples.append({
                            "instruction": instruction,
                            "input": "",
                            "output": output
                        })
            except Exception as e:
                print(f"⚠️ Error procesando línea: {line[:50]}... - {e}")
                continue
    
    return examples


def save_jsonl(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def main():
    print(f">>> Leyendo {RAW_FILE.name}...")
    if not RAW_FILE.exists():
        print(f"❌ No se encontró el archivo: {RAW_FILE}")
        return

    raw_text = RAW_FILE.read_text(encoding="utf-8")

    # Detectar formato y procesar según corresponda
    if RAW_FILE.suffix == ".md":
        print(">>> Formato detectado: Markdown (### Instrucción / ### Respuesta)")
        examples = parse_markdown(raw_text)
    else:
        print(">>> Formato detectado: Conversaciones (ALUMNO: / TUTOR:)")
        examples = parse_conversaciones(raw_text)
    
    print(f">>> Ejemplos encontrados: {len(examples)}")

    if len(examples) == 0:
        print("❌ No se encontraron ejemplos. Revisa el formato del archivo.")
        return
    
    if len(examples) < 50:
        print("⚠️ Se recomienda al menos 100 ejemplos para mejores resultados.")

    random.shuffle(examples)

    total = len(examples)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_data = examples[:train_end]
    val_data = examples[train_end:val_end]
    test_data = examples[val_end:]

    print(f">>> Train: {len(train_data)}")
    print(f">>> Val: {len(val_data)}")
    print(f">>> Test: {len(test_data)}")

    save_jsonl(train_data, TRAIN_FILE)
    save_jsonl(val_data, VAL_FILE)
    save_jsonl(test_data, TEST_FILE)

    print("✅ Preprocesamiento completado")
    print(f"Archivos guardados en {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
