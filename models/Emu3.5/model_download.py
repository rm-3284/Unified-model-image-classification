from huggingface_hub import snapshot_download

snapshot_download(repo_id="BAAI/Emu3.5-Image", cache_dir="/n/fs/vision-mix/rm4411/Emu3.5/weights")

snapshot_download(repo_id="BAAI/Emu3.5-VisionTokenizer", cache_dir="/n/fs/vision-mix/rm4411/Emu3.5/weights")
