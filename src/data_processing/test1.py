from pathlib import Path
from PIL import Image
for d in sorted(Path('data/realbook_primus').iterdir()):
    sem = d / f'{d.name}.semantic'
    png = d / f'{d.name}.png'
    tokens = len(sem.read_text().split()) if sem.exists() else 0
    size = Image.open(png).size if png.exists() else None
    print(f'{d.name}: {tokens} tokens -> {size}')
