
For Missrate when not multiple_outputs is not turned on, note that missrate results are saved as json file to the runs/test/foldname/ as miss_rate_{filename}.json (for the Kaist Dataset)

#ct2 stands for confidence thereshold at 0.2
python ./miss_rate_and_map/evaluation_script.py \
    --annFile miss_rate_and_map/KAIST_annotation.json \
    --rstFiles ./runs/test/foldername/best_predictions_ct2.json 

#ct001 stands for confidence thereshold at 0.001
python ./miss_rate_and_map/evaluation_script.py \
--annFile miss_rate_and_map/KAIST_annotation.json\
--rstFiles ./runs/test/foldername/best_predictions_ct001.json


For mAP, not that this runs and saves the overrall mAP and class level map into a json file to the folder runs/test/foldername as mAP_mAR_{filename}.json

   python ./miss_rate_and_map/map_calc.py \
    --annFile miss_rate_and_map/kaist_test20_for_map.json \
    --rstFiles ./runs/test/foldername/best_predictions_ct001_conf.json


   python ./miss_rate_and_map/map_calc.py \
    --annFile miss_rate_and_map/kaist_test20_for_map.json \
    --rstFiles ./runs/test/foldername/best_predictions_ct2_conf.json