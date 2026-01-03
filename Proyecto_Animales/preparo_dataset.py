import argparse
import math
import os
from pathlib import Path
from icrawler.builtin import BingImageCrawler

# Etiquetas y términos de búsqueda (en inglés y español)
CLASS_QUERIES = {
    "dogs": {
        "queries": ["dog photo", "perro", "pet dog"],
        "max_num": 3000,
    },
    "cats": {
        "queries": ["cat photo", "gato", "pet cat"],
        "max_num": 3000,
    },
    "ladybugs": {
        "queries": ["ladybug", "lady bird", "mariquita", "vaquita de San Antonio", "coccinellidae"],
        "max_num": 3000,
    },
    "ants": {
        "queries": ["ant", "formicidae", "hormiga"],
        "max_num": 3000,
    },
    "turtles": {
        "queries": ["turtle", "tortoise", "galapagos turtle", "tortuga"],
        "max_num": 3000,
    },
}

# Filtros de Bing: solo fotos con licencia creative commons
BING_FILTERS = {
    "type": "photo",
    "license": "creativecommons",
}


def crawl_class(label: str, queries: list[str], max_num: int, output_root: Path) -> None:
    class_dir = output_root / label
    class_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(class_dir.glob("*")))
    remaining = max(0, max_num - existing)
    if remaining <= 0:
        print(f"[SKIP] {label}: ya hay {existing} imágenes (>= {max_num}).")
        return

    per_query = math.ceil(remaining / len(queries))
    idx_offset = existing

    for q in queries:
        # Offset evita sobreescritura entre consultas
        crawler = BingImageCrawler(
            storage={"root_dir": str(class_dir)},
            downloader_threads=4,
            feeder_threads=1,
            parser_threads=2,
        )
        crawler.crawl(
            keyword=q,
            filters=BING_FILTERS,
            max_num=per_query,
            file_idx_offset=idx_offset,
        )
        idx_offset += per_query
        print(f"[DONE] {label} <- {q}")


def main():
    parser = argparse.ArgumentParser(description="Descarga imágenes por clase usando Bing (Creative Commons)")
    parser.add_argument(
        "--out",
        default=os.path.join(os.getcwd(), "CNN_ejemplo", "animals-dataset"),
        help="Directorio raíz donde se guardarán las imágenes (default: ./CNN_ejemplo/animals-dataset)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Máximo de imágenes por clase (sobrescribe los defaults del diccionario)",
    )
    args = parser.parse_args()

    output_root = Path(args.out).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for label, cfg in CLASS_QUERIES.items():
        target_max = args.max if args.max is not None else cfg["max_num"]
        crawl_class(label, cfg["queries"], target_max, output_root)

    print(f"Descarga completa. Revisa {output_root}.")


if __name__ == "__main__":
    main()
