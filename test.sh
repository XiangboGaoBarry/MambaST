
CUDA_VISIBLE_DEVICES=0 \
python test_video.py \
    --weights ./runs/train/vim_411_ocf_uniscan/weights/last.pt \
    --data ./data/multispectral_temporal/kaist_video_test.yaml \
    --name vim_411_ocf_uniscan \
    --task test \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --regex_search .set...V... \
    --dataset_used kaist \
    --img-size 640 \
    --save-json \
    --exist-ok \
    --sanitized \
    --conf-thres 0.001 \
    --batch-size 32 \
    --iou-thres 0.5 \
    --detector_weights both \

