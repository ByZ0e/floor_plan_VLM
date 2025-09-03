import re
import os
import json
import traceback
from tqdm import tqdm
import json
import os
import glob
import supervision as sv
import numpy as np
import cv2
import torch
import traceback
import argparse
from torchvision.ops import box_convert
import sys
sys.path.append(os.path.abspath('..'))
# import call_gpt
# from utils import call_gpt
from call_gpt import request_gpt4

from groundingdino.util.inference import load_model, load_image, predict, annotate

prompt = '''
# Instruction
Given the task goal, tell me what are the most related objects or landmarks during handling this task and output them. As concise as possible and they have to be single objects. You SHOULD return less than 10 key objects.

# Example
"Input (Task goal)":
Go to robot training ground 1 and robot training ground 3 from kitchen
"Output":
[fridge, sink, robot, hallway, table, desk, cabinet, chair, staff]

Now, please read the following task goal and return less then 10 key objects.
"Input (Task goal)":
{goal}
"Output":
'''

def key_objs(samples, objs_path):
    save_dict = {}
    for sample in tqdm(samples):
        try:
            instruction = prompt.format(goal=sample["task_goal"])
        
            message_content = [{"type": "text", "text": instruction}]
            response = request_gpt4(message_content)

            matches = re.findall(r'\[(.*?)\]', response)[0].split(',')
            objects = [x.strip() for x in matches]
            save_dict[sample["sample_id"]] = objects
            with open(objs_path, 'w') as f:
                json.dump(save_dict, f, indent=4)

        except Exception:
            print(f"Error occurs when processing this query: {sample['sample_id']}", flush=True)
            traceback.print_exc()  

def my_annotate(
    image_source: np.ndarray,
    boxes: torch.Tensor,
    logits: torch.Tensor,
    phrases,
) -> np.ndarray:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)

    for box, logit, phrase in zip(xyxy, logits, phrases):
        x1, y1, x2, y2 = map(int, box)
        label = f"{phrase} {logit:.2f}"

        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1) # green
        
        # # Draw label background box
        # (text_width, text_height), _ = cv2.getTextSize(
        #     label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        # )
        # cv2.rectangle(
        #     annotated_frame,
        #     (x1, y1 - text_height - 4),
        #     (x1 + text_width, y1),
        #     (0, 255, 0),
        #     -1,
        # )

        # # Draw label text
        # cv2.putText(
        #     annotated_frame,
        #     label,
        #     (x1, y1 - 2),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (0, 0, 0),
        #     1,
        # )

    return annotated_frame

def bounding_boxs(samples, objs_path, dino_dir, keyframes_dir, bbox_dir, obj_num=1):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # load DINO
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25
    model = load_model(os.path.join(dino_dir, "groundingdino/config/GroundingDINO_SwinT_OGC.py"), 
                       os.path.join(dino_dir, "weights/groundingdino_swint_ogc.pth"))
    
    key_objs = json.load(open(objs_path))

    if not os.path.exists(bbox_dir): os.makedirs(bbox_dir)

    for sample in samples:
        qa_id = sample["sample_id"]
        
        if qa_id not in key_objs.keys():
            continue
        # if os.path.exists(os.path.join(bbox_dir, qa_id+'.jpg')):
        #     continue
        try:
            if obj_num == -1:
                # TEXT_PROMPT = "human hand . "
                obj_list = key_objs[qa_id]
                TEXT_PROMPT = " . ".join(obj_list) + ' .'
                TEXT_PROMPT = TEXT_PROMPT.strip()
                IMAGE_PATH = glob.glob(os.path.join(keyframes_dir, qa_id, f'frameID-*.png'))[0]
                print(f"\n{TEXT_PROMPT}")

                image_source, image = load_image(IMAGE_PATH)

                boxes, logits, phrases = predict(
                    model=model,
                    image=image,
                    caption=TEXT_PROMPT,
                    box_threshold=BOX_TRESHOLD,
                    text_threshold=TEXT_TRESHOLD,
                    device=DEVICE
                )
                
                annotated_frame = my_annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
                cv2.imwrite(os.path.join(bbox_dir, qa_id+'.jpg'), annotated_frame)

                boxes = boxes.numpy().flatten().tolist()
                logits = logits.numpy().tolist()
                save_dict = {
                    "boxes": boxes,
                    "logits": logits, 
                    "phrases": phrases
                }
                with open(os.path.join(bbox_dir, qa_id+'.json'), 'w') as f:
                    json.dump(save_dict, f, indent=4)

            else:
                IMAGE_PATH = glob.glob(os.path.join(keyframes_dir, qa_id, f'frameID-*.png'))[0]

                best_boxes = []
                best_phrases = []
                best_logits = []
                
                obj_list = key_objs[qa_id]
                for obj in obj_list:
                    TEXT_PROMPT = obj.strip() + ' .'
                    print(f"\n{TEXT_PROMPT}")

                    image_source, image = load_image(IMAGE_PATH)

                    boxes, logits, phrases = predict(
                        model=model,
                        image=image,
                        caption=TEXT_PROMPT,
                        box_threshold=BOX_TRESHOLD,
                        text_threshold=TEXT_TRESHOLD,
                        device=DEVICE
                    )

                    if boxes.shape[0] > 0:
                        selected_count = min(obj_num, boxes.shape[0])  # If returned boxes are fewer than object count
                        for i in range(selected_count):
                            best_boxes.append(boxes[i].unsqueeze(0))
                            best_phrases.append(phrases[i])
                            best_logits.append(logits[i])
                
                if best_boxes:
                    best_boxes = torch.cat(best_boxes)
                    best_logits = torch.stack(best_logits)
                
                annotated_frame = my_annotate(image_source=image_source, boxes=best_boxes, logits=best_logits, phrases=best_phrases)
                cv2.imwrite(os.path.join(bbox_dir, qa_id+'.jpg'), annotated_frame)

                best_boxes = best_boxes.numpy().flatten().tolist()
                best_logits = best_logits.numpy().tolist()
                save_dict = {
                    "boxes": best_boxes,
                    "logits": best_logits, 
                    "phrases": best_phrases
                }
                with open(os.path.join(bbox_dir, qa_id+'.json'), 'w') as f:
                    json.dump(save_dict, f, indent=4)

        except Exception:
            print(f"Error occurs when processing this query: {qa_id}", flush=True)
            traceback.print_exc()   

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arg Parser')
    parser.add_argument('--anno_path', type=str, default='data/BAAI_navi/BAAI_navi.json')
    parser.add_argument('--dino_dir', type=str, default='GroundingDINO')
    parser.add_argument('--keyframes_dir', type=str, default='visual/keyframes_dir')
    parser.add_argument('--objs_path', type=str, default='visual/cot/key_objs.json')
    parser.add_argument('--bbox_dir', type=str, default='visual/cot/bounding_box')
    args = parser.parse_args()

    anno_path = args.anno_path
    objs_path = args.objs_path
    dino_dir = args.dino_dir
    keyframes_dir = args.keyframes_dir
    bbox_dir = args.bbox_dir
    samples = json.load(open(args.anno_path))

    if not os.path.exists('visual/cot'): os.makedirs('visual/cot')
    
    # key_objs(samples, objs_path)
    bounding_boxs(samples, objs_path, dino_dir, keyframes_dir, bbox_dir, obj_num=1)
