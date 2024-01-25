
import torch
import torch.utils.data
import json
from box_ops import generalized_box_iou, box_iou

class RefExpEvaluatorFromTxt(object):
    def __init__(self, gt_pred_path, thresh_iou=0.5):
        with open(gt_pred_path, 'r') as f:
            self.grounding = json.load(f)
        print(f"Load {len(self.grounding)} images")
        self.thresh_iou = thresh_iou

    def summarize(self,
                  quantized_size: int = 32,
                  verbose: bool = False,):

        max_iou_match_sum = 0
        cnt_match = 0
        cnt_sum = 0
        cnt_succ=0
        cnt_succ_sum=0
        max_iou_sum = 0
        for g in self.grounding:
            gt_box=g['gt']
            gt_box = [gt_box[0], gt_box[1], gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]]
            out_box=g['out_boxes']
            out_box_0 = out_box
            out_box_0 = torch.as_tensor(out_box_0, dtype=torch.float).tolist()
            target_bbox = torch.as_tensor(gt_box).view(-1, 4)
            predict_boxes=[]
            predict_boxes.append(out_box_0)

            if len(predict_boxes) == 0:
                print(f"Can't find valid bbox for the given phrase")
                print(f"We set a 0-area box to calculate result")
                predict_boxes = [[0., 0., 0., 0.]]

            predict_boxes = torch.as_tensor(predict_boxes).view(-1, 4)

            iou, _ = box_iou(predict_boxes, target_bbox)
            mean_iou, _ = box_iou(predict_boxes.mean(0).view(-1, 4), target_bbox)
            max_iou_sum += max(iou)
            cnt_sum += 1
            if max(iou) >= self.thresh_iou:
                cnt_succ += 1.0
                max_iou_match_sum += max(iou)
                cnt_match += 1

            cnt_succ_sum += 1.0

        print('succ_rate: ',cnt_succ/cnt_succ_sum)
        print('miou: ', max_iou_sum / cnt_sum)
        print('mean_iou(match): ', max_iou_match_sum / cnt_match)
        return 0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_predict_file',default='/path/to/predict_grounding.json',help='gt_prediction_file')
    parser.add_argument('--quantized_size', default=32, type=int)
    args = parser.parse_args()
    
    evaluator = RefExpEvaluatorFromTxt(
        gt_pred_path=args.gt_predict_file,
        thresh_iou=0.5,
    )
    
    evaluator.summarize(args.quantized_size, verbose=False)
