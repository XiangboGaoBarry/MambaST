CUDA_VISIBLE_DEVICES='0' \
python train_video.py \
    --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3_whole_nc1.yaml \
    --batch-size 2 \
    --epochs 40 \
    --cfg ./models/transformer/two_detection_heads_v2/vim/yolov5l_transformerx3_Kaist_vimV2_nc1_411_uniscan_lframe3.yaml \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name vim_411_ocf_uniscan \
    --exist-ok \
    --hyp data/hyp.finetune_focal_loss_high_obj_low_scale.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --sanitized \
    --detector_weights both \
    --thermal_weights yolov5l_kaist_best_thermal.pt \
    --rgb_weights yolov5l_kaist_best_rgb.pt \
    --temporal_mosaic \
    --mosaic \
    --noautoanchor

