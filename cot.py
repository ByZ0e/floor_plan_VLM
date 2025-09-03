import os
import re
import csv
import json
import glob
import argparse

from call_gpt import request_base64_gpt4v, encode_image

system_prompt='''
I am a robot that cannot go through walls and must use doors or openings to traverse through different areas to accomplish the task.
I rely on my first-person view observation (CURRENT OBSERVATION) to identify real-world elements like objects, doors, openings, and pathways, which are essential for guidin my actions. 
I can also consult a floor plan (FLOORMAP), a top-down representation of the building to understand the overall layout of rooms.
I need to output specific movement actions based solely on what I see in the observation, using the floormap only for general contextual information.
'''

incontext_prompt='''
# INPUT
[CURRENT OBSERVATION]: The current observation image with bounding boxes which outline key objects that may be relevant to the task. 
[FLOORMAP]: The top-down floor plan image, showing rooms, doors, walls, and pathways (green areas are corridors, blue rectangles are doors/opens)
[TASK]: The task to accomplish.

# OUTPUT FORMAT
1. [CURRENT OBSERVATION]: Briefly describe what you see in the image, focusing on key objects, doors, openings, turning corners, navigatable pathways or any other landmarks. Infer your current location by matching with FLOORMAP.
2. [PATH CONTEXT]: Using FLOORMAP, identify the general path to the goal (e.g., which rooms or halls to traverse).
3. [NEXT ACTION]: Output the immediate next action (e.g., \"Turn right to face the coffee machine\", \"Go straight to the white table\") as a specific command. Do not rely on floormap symbols or concepts, such as \"green corridors\"; use only what is visible in your observations, such as \"white table\". Do NOT use directions like 'North' or 'East' with the floormap coordinate. Must use relative directions like 'left', 'right', 'forward' based on your current orientation.

Now, please analyze the following case:
[CURRENT OBSERVATION]
<observation>
[FLOORMAP]
<floormap>
[TASK]
What action should you take next to {goal}?
'''

def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:"
        "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ""
    return matches[0]

def gpt4v_interleave(instruction, images_path, extra_images_path = None, system_prompt = None, seed = 42):

    content_images = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(x)}", "detail": "high"},
        } for x in images_path
    ]

    if extra_images_path:
        content_images_extra = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(x)}", "detail": "high"},
            } for x in extra_images_path
        ]
    
    if system_prompt:
        system_content = [{"type": "text", "text": system_prompt}]
    else:
        system_content = None

    texts = re.split('<observation>|<floormap>', instruction)
    message_content = [{"type": "text", "text": texts[0]}] + content_images + [{"type": "text", "text": texts[1]}] + content_images_extra + [{"type": "text", "text": texts[2]}]
    
    # print(f"SYSTEM:\n{system_content}\nUSER:\n")
    # for x in message_content:
    #     if x["type"] == "text":
    #         print(x)
    #     else:
    #         print("<image>")
    
    response = request_base64_gpt4v(message_content, system_content, seed)
    return response

def run_inference(qa_anno, bbox_dir, output_dir, seed):
    # count, correct = 0, 0
    output_f = open(os.path.join(output_dir), "a")

    for qa_item in qa_anno:
        qa_id = qa_item['sample_id']
        
        # multimodal prompts
        # history = json.load(open(f"{cap_dir}/{qa_id}.json"))["Caption"]
        # images_path = [glob.glob(os.path.join(keyframes_dir, qa_id, f'frameID-*.png'))[0]]
        images_path = [f"{bbox_dir}/{qa_id}.jpg"]
        obj_images_path = [f"BAAI_semantic_floor_map_door.png"]
        print(f"\nimage paths: {images_path}")
        print(f"\nextra image paths: {obj_images_path}")

        instruction = incontext_prompt.format(
            # history = history,
            goal = qa_item['task_goal'],
            # choice_a = qa_item['choice_a'],
            # choice_b = qa_item['choice_b'],
            # choice_c = qa_item['choice_c'],
            # choice_d = qa_item['choice_d']
        )

        response = gpt4v_interleave(instruction, images_path, extra_images_path = obj_images_path, system_prompt = system_prompt, seed = seed)
        # extraction = extract_characters_regex(re.split('\[ANSWER\]:', response)[-1]).upper()
        print(f"\nmodel response: {response}")

        # count += 1
        # correct += extraction == qa_item['golden_choice_idx']
        # print(f"\nQA nums: {count}, correct: {correct}, acc: {correct/count}\n")

        answer_record = {
            "sample_id": qa_id,
            # "ground_truth": qa_item['golden_choice_idx'],
            "model_response": response,
            # "extracted_answer": extraction,
            # "count": count,
            # "correct": correct,
            # "acc": correct/count
        }
        output_f.write(json.dumps(answer_record) + "\n")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arg Parser')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--anno_path', type=str, default='data/BAAI_navi/BAAI_navi.json')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--visual_input_dir', type=str, default='visual')
    args = parser.parse_args()


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    import time
    qa_anno = json.load(open(args.anno_path))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f'CoT_GPT-4o_{timestamp}.json')
    # keyframes_dir = os.path.join(args.visual_input_dir, 'keyframes_dir')
    # cap_dir = os.path.join(args.visual_input_dir, 'cot', 'cap')
    bbox_dir = os.path.join(args.visual_input_dir, 'cot', 'bounding_box')


    run_inference(qa_anno, bbox_dir, output_dir, args.seed)