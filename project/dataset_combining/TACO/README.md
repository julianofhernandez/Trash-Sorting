## Steps for download
In CMD:
```
git clone https://github.com/pedropro/TACO.git
python37 -m pip install -r requirements.txt
python37 download.py
```

## Issues
Trying to download the extra 3700 images that arene't official results in an erros because the system cannot handle RGBD images yet.
```
python3 download.py --dataset_path ./data/annotations_unofficial.json
```