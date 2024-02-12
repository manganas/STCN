import os
import gdown
import zipfile
from scripts import resize_youtube


LICENSE = """
These are either re-distribution of the original datasets or derivatives (through simple processing) of the original datasets. 
Please read and respect their licenses and terms before use. 
You should cite the original papers if you use any of the datasets.

For BL30K, see download_bl30k.py

Links:
DUTS: http://saliencydetection.net/duts
HRSOD: https://github.com/yi94code/HRSOD
FSS: https://github.com/HKUSTCV/FSS-1000
ECSSD: https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html
BIG: https://github.com/hkchengrex/CascadePSP

YouTubeVOS: https://youtube-vos.org
DAVIS: https://davischallenge.org/
BL30K: https://github.com/hkchengrex/MiVOS
"""

print(LICENSE)
print(
    "Datasets will be downloaded and extracted to ../YouTube, ../YouTube2018, ../static, ../DAVIS"
)
reply = input("[y] to confirm, others to exit: ")
if reply != "y":
    exit()


BASE_DIR = r"/work3/s220493"

# Static data
os.makedirs(BASE_DIR + "/static", exist_ok=True)
print("Downloading static datasets...")
gdown.download(
    "https://drive.google.com/uc?id=1wUJq3HcLdN-z1t4CsUhjeZ9BVDb9YKLd",
    output=BASE_DIR + "/static/static_data.zip",
    quiet=False,
)
print("Extracting static datasets...")
with zipfile.ZipFile(BASE_DIR + "/static/static_data.zip", "r") as zip_file:
    zip_file.extractall(BASE_DIR + "/static/")
print("Cleaning up static datasets...")
os.remove(BASE_DIR + "/static/static_data.zip")


# DAVIS
# Google drive mirror: https://drive.google.com/drive/folders/1hEczGHw7qcMScbCJukZsoOW4Q9byx16A?usp=sharing
os.makedirs(BASE_DIR + "/DAVIS/2017", exist_ok=True)

print("Downloading DAVIS 2016...")
gdown.download(
    "https://drive.google.com/uc?id=198aRlh5CpAoFz0hfRgYbiNenn_K8DxWD",
    output=BASE_DIR + "/DAVIS/DAVIS-data.zip",
    quiet=False,
)

print("Downloading DAVIS 2017 trainval...")
gdown.download(
    "https://drive.google.com/uc?id=1kiaxrX_4GuW6NmiVuKGSGVoKGWjOdp6d",
    output=BASE_DIR + "/DAVIS/2017/DAVIS-2017-trainval-480p.zip",
    quiet=False,
)

print("Downloading DAVIS 2017 testdev...")
gdown.download(
    "https://drive.google.com/uc?id=1fmkxU2v9cQwyb62Tj1xFDdh2p4kDsUzD",
    output=BASE_DIR + "/DAVIS/2017/DAVIS-2017-test-dev-480p.zip",
    quiet=False,
)

print("Downloading DAVIS 2017 scribbles...")
gdown.download(
    "https://drive.google.com/uc?id=1JzIQSu36h7dVM8q0VoE4oZJwBXvrZlkl",
    output=BASE_DIR + "/DAVIS/2017/DAVIS-2017-scribbles-trainval.zip",
    quiet=False,
)

print("Extracting DAVIS datasets...")
with zipfile.ZipFile(BASE_DIR + "/DAVIS/DAVIS-data.zip", "r") as zip_file:
    zip_file.extractall(BASE_DIR + "/DAVIS/")
os.rename(BASE_DIR + "/DAVIS/DAVIS", BASE_DIR + "/DAVIS/2016")

with zipfile.ZipFile(
    BASE_DIR + "/DAVIS/2017/DAVIS-2017-trainval-480p.zip", "r"
) as zip_file:
    zip_file.extractall(BASE_DIR + "/DAVIS/2017/")
with zipfile.ZipFile(
    BASE_DIR + "/DAVIS/2017/DAVIS-2017-scribbles-trainval.zip", "r"
) as zip_file:
    zip_file.extractall(BASE_DIR + "/DAVIS/2017/")
os.rename(BASE_DIR + "/DAVIS/2017/DAVIS", BASE_DIR + "/DAVIS/2017/trainval")

with zipfile.ZipFile(
    BASE_DIR + "/DAVIS/2017/DAVIS-2017-test-dev-480p.zip", "r"
) as zip_file:
    zip_file.extractall(BASE_DIR + "/DAVIS/2017/")
os.rename(BASE_DIR + "/DAVIS/2017/DAVIS", BASE_DIR + "/DAVIS/2017/test-dev")

print("Cleaning up DAVIS datasets...")
os.remove(BASE_DIR + "/DAVIS/2017/DAVIS-2017-trainval-480p.zip")
os.remove(BASE_DIR + "/DAVIS/2017/DAVIS-2017-test-dev-480p.zip")
os.remove(BASE_DIR + "/DAVIS/2017/DAVIS-2017-scribbles-trainval.zip")
os.remove(BASE_DIR + "/DAVIS/DAVIS-data.zip")


# YouTubeVOS
os.makedirs(BASE_DIR + "/YouTube", exist_ok=True)
os.makedirs(BASE_DIR + "/YouTube/all_frames", exist_ok=True)

print("Downloading YouTubeVOS train...")
gdown.download(
    "https://drive.google.com/uc?id=13Eqw0gVK-AO5B-cqvJ203mZ2vzWck9s4",
    output=BASE_DIR + "/YouTube/train.zip",
    quiet=False,
)
print("Downloading YouTubeVOS val...")
gdown.download(
    "https://drive.google.com/uc?id=1o586Wjya-f2ohxYf9C1RlRH-gkrzGS8t",
    output=BASE_DIR + "/YouTube/valid.zip",
    quiet=False,
)
print("Downloading YouTubeVOS all frames valid...")
gdown.download(
    "https://drive.google.com/uc?id=1rWQzZcMskgpEQOZdJPJ7eTmLCBEIIpEN",
    output=BASE_DIR + "/YouTube/all_frames/valid.zip",
    quiet=False,
)

print("Extracting YouTube datasets...")
with zipfile.ZipFile(BASE_DIR + "/YouTube/train.zip", "r") as zip_file:
    zip_file.extractall(BASE_DIR + "/YouTube/")
with zipfile.ZipFile(BASE_DIR + "/YouTube/valid.zip", "r") as zip_file:
    zip_file.extractall(BASE_DIR + "/YouTube/")
with zipfile.ZipFile(BASE_DIR + "/YouTube/all_frames/valid.zip", "r") as zip_file:
    zip_file.extractall(BASE_DIR + "/YouTube/all_frames")

print("Cleaning up YouTubeVOS datasets...")
os.remove(BASE_DIR + "/YouTube/train.zip")
os.remove(BASE_DIR + "/YouTube/valid.zip")
os.remove(BASE_DIR + "/YouTube/all_frames/valid.zip")

print("Resizing YouTubeVOS to 480p...")
resize_youtube.resize_all(BASE_DIR + "/YouTube/train", BASE_DIR + "/YouTube/train_480p")

# YouTubeVOS 2018
os.makedirs(BASE_DIR + "/YouTube2018", exist_ok=True)
os.makedirs(BASE_DIR + "/YouTube2018/all_frames", exist_ok=True)

print("Downloading YouTubeVOS2018 val...")
gdown.download(
    "https://drive.google.com/uc?id=1-QrceIl5sUNTKz7Iq0UsWC6NLZq7girr",
    output=BASE_DIR + "/YouTube2018/valid.zip",
    quiet=False,
)
print("Downloading YouTubeVOS2018 all frames valid...")
gdown.download(
    "https://drive.google.com/uc?id=1yVoHM6zgdcL348cFpolFcEl4IC1gorbV",
    output=BASE_DIR + "/YouTube2018/all_frames/valid.zip",
    quiet=False,
)

print("Extracting YouTube2018 datasets...")
with zipfile.ZipFile(BASE_DIR + "/YouTube2018/valid.zip", "r") as zip_file:
    zip_file.extractall(BASE_DIR + "/YouTube2018/")
with zipfile.ZipFile(BASE_DIR + "/YouTube2018/all_frames/valid.zip", "r") as zip_file:
    zip_file.extractall(BASE_DIR + "/YouTube2018/all_frames")

print("Cleaning up YouTubeVOS2018 datasets...")
os.remove(BASE_DIR + "/YouTube2018/valid.zip")
os.remove(BASE_DIR + "/YouTube2018/all_frames/valid.zip")

print("Done.")
