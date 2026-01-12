import os
import json


def map_directory_fast(root_path):
    file_map = {}
    root_path = os.path.abspath(root_path)

    print("Scanne Verzeichnis... (bitte warten)")
    for root, dirs, files in os.walk(root_path):
        for name in files:
            full_path = os.path.join(root, name)
            rel_path = os.path.relpath(full_path, root_path)
            try:
                stats = os.stat(full_path)
                # Wir speichern nur Größe und Änderungsdatum
                file_map[rel_path] = (stats.st_size, stats.st_mtime)
            except OSError:
                continue  # Überspringe Dateien, auf die kein Zugriff besteht
    return file_map


source_dir = input("Pfad zum Quell-Ordner: ")
output_file = "fast_file_map.json"

mapping = map_directory_fast(source_dir)
with open(output_file, 'w') as f:
    json.dump(mapping, f)

print(f"Fertig! {len(mapping)} Dateien erfasst.")