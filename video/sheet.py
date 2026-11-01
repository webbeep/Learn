import sys, glob
from PIL import Image
files = sorted(glob.glob(sys.argv[1] + "/*_boxes.png")) if len(sys.argv) < 4 else sys.argv[3:]
cols = 3; w = int(sys.argv[2]) if len(sys.argv) > 2 else 640
ims = [Image.open(f) for f in files]; h = int(w * 9 / 16)
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (cols * w + (cols - 1) * 6, rows * h + (rows - 1) * 6), (60, 60, 60))
for i, im in enumerate(ims):
    sheet.paste(im.resize((w, h)), ((i % cols) * (w + 6), (i // cols) * (h + 6)))
sheet.save(sys.argv[1] + "/sheet.png"); print(sys.argv[1] + "/sheet.png")
