

CUDA_VISIBLE_DEVICES='0' \
python kaist_to_json.py \
    --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3_whole_nc1.yaml \
    --batch-size 8 \
    --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned.yaml \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search .set...V... \
    --temporal_mosaic \
    --image_set train \
    --all_objects \
    --ignore_high_occ



