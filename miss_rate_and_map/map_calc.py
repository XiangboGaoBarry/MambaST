from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse
from pathlib import Path
from collections import defaultdict
import json

def mAP_calc(anno_json_path, detect_json_path, catIds=None):
    anno = COCO(anno_json_path)  # init annotations api
    pred = anno.loadRes(detect_json_path)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')

    if catIds is not None:
        eval.params.catIds = catIds

    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    # print('*'*40)
    # print(eval.stats) #ap_0.5:0.95_all, ap_0.5_all, ap_0.75_all, ap_0.5:0.95_small, ap_0.5:0.95_med, ap_0.5:0.95_large
    # # #ap_0.5:0.95_all
    # map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)

    # print(map)
    return eval.stats

def map_json_file(anno_json, detection_json, dataset_used='kaist', multiple_outputs=True):
    """
    Use to calcuate multiple mAP
    """
    # Need to loop through different pred_json
    # For each pred_json, need to loop through the different classes
    json_dict = defaultdict(list)
    result_txt = ['mAP_0.5:0.95', 'mAP_0.5', 'mAP_0.75',
                  'mAP_0.5:0.95_small', 'mAP_0.5:0.95_med', 'mAP_0.5:0.95_large',
                  'mAR_0.5:0.95_maxdet_1', 'mAR_0.5:0.95_maxdet_10', 'mAR_0.5:0.95',
                  'mAR_0.5:0.95_small', 'mAR_0.5:0.95_med', 'mAR_0.5:0.95_large']
    if dataset_used == 'kaist':
        obj_map = {1:'person', 2: 'people', 3:'cyclist', 4:"person?" }
    
    path = Path(detection_json)
    spath = str(path.parent)
    if multiple_outputs:
        rstFile_list = []
        for i in range(0,40):
            temp = f'cur_{i}_predictions_conf.json'
            rstFile_list.append(spath + '/' + temp)
    else:
        rstFile_list = [detection_json]
    for rstfile in rstFile_list:
        if dataset_used == 'kaist':
            for catID in [None, 1,2,3,4]:
                results = mAP_calc(anno_json, rstfile, catIds=catID)
                for i, result in enumerate(results):
                    if i == 6 or i == 7:
                        # skip 'mAR_0.5:0.95_maxdet_1' and 'mAR_0.5:0.95_maxdet_10'
                        continue
                    if catID is None:
                        key_elem = result_txt[i]
                    else:
                        key_elem = result_txt[i] + '_' + obj_map[catID]
                    
                    json_dict[key_elem].append(result)
                
        else:
            raise 'Not Kaist Dataset'
    if multiple_outputs:
        save_loc = spath + '/' + 'mAP_mAR.json'
    else:
        save_loc = spath + '/' + f'mAP_mAR_{path.name}'
    with open(save_loc, 'w') as f:
        json.dump(json_dict, f)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval models')
    parser.add_argument('--annFile', type=str, default='./json_gt/kaist_test20_for_map.json',
                        help='Please put the path of the annotation file. Only support json format.')
    parser.add_argument('--rstFiles', type=str, nargs='+', default=['evaluation_script/MLPD_result.json'],
                        help='Please put the path of the result file. Only support json, txt format.')
    parser.add_argument('--evalFig', type=str, default='KASIT_BENCHMARK.jpg',
                        help='Please put the output path of the Miss rate versus false positive per-image (FPPI) curve')
    parser.add_argument('--multiple_outputs', action='store_true', help='evaluate muliple json and save result')
    parser.add_argument('--dataset_used', type=str, default="kaist", help='dataset used: kaist, camel,')
    args = parser.parse_args()

    anno_json = args.annFile
    detection_json = args.rstFiles[0]
    # anno_json = './json_gt/kaist_test20_for_map.json'
    # pred_json = './runs/test/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialGPT_lastframe_backbone_head_loaded2_3_stride_3_conf_thres_2_delete_later_mid_point3/best_predictions.json'
    # anno_json = './miss_rate_and_map/KAIST_annotation.json'
    # pred_json = './runs/test/fusion_transformerx3_kaist_video_backbone_head_loaded_lframe_one_stride_one2/best_predictions_conf.json'
    # detection_json = './runs/test/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_lastframe_thermal_rgb_backbone_head_thermal_lframe_three_stride_three3/cur_30_predictions_conf.json'
    map_json_file(anno_json, detection_json, args.dataset_used, args.multiple_outputs)
    # mAP_calc(anno_json, detection_json)