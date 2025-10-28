
import gdown, os
out = os.path.join(os.path.dirname(__file__), os.pardir, "data", "heart.csv")
out = os.path.abspath(out)
# Google Drive file id (from report)
file_id = "1U5Iwn7X_oJWmSiYBSF2AuQdiEvIaUMXv"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
print("Downloading dataset to", out)
gdown.download(url, out, quiet=False)
print("Done. If download failed, verify the Drive file's share settings.")
