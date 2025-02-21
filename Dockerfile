# Use PyTorch 2.1.2 CUDA base image
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt && rm -rf /root/.cache/pip/*

COPY dataset/ ./dataset/
COPY trainer_misc/ ./trainer_misc/
COPY video_vae/ ./video_vae/
COPY utils.py .

COPY scripts/train_causal_video_vae.sh .
COPY train/train_video_vae.py ./train/
COPY pyramid_flow_model/causal_video_vae/ ./pyramid_flow_model/causal_video_vae/
COPY vgg_lpips.pth .

RUN chmod +x train_causal_video_vae.sh

ENV OUTPUT_DIR=/mnt/ssd/experiments/Pyramid-Flow/output_dir
ENV IMAGE_ANNO=/mnt/ssd/datasets/processed_data/image.jsonl
ENV VIDEO_ANNO=/mnt/ssd/datasets/processed_data/video.jsonl

RUN mkdir -p $OUTPUT_DIR

CMD ["./train_causal_video_vae.sh"]