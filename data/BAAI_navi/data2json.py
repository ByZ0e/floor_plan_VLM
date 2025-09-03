import re
import json

def parse_task_goal(sample_id):
    # 解析起点和终点
    parts = sample_id.split('_')
    if len(parts) >= 3:
        start = parts[0][1:]
        start = start.replace('RTG', 'robot training ground ')
        start = start.replace('meeting', 'meeting room ')
        start = start.replace('boye', 'workspace')

        goal = parts[1]
        goal = goal.replace('RTG', 'robot training ground ')
        goal = goal.replace('meeting', 'meeting room ')
        goal = goal.replace('boye', 'workspace')
        return f"Go to {goal} from {start}"
    return "Unknown task goal"

def time_to_frame(time_str, fps=24):
    # 时间格式 mm:ss
    m, s = map(int, time_str.split(':'))
    return (m * 60 + s) * fps

def process_file(input_path, output_path, fps=24):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results = []
    sample_id = None
    metadata = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 检查是否为sample_id
        if re.match(r'^[\w\d]+_.*_L$', line):
            if sample_id and metadata:
                results.append({
                    "sample_id": sample_id,
                    "task_goal": parse_task_goal(sample_id),
                    "task_progress_metadata": metadata
                })
            sample_id = line
            metadata = []
        else:
            # 匹配时间和描述
            match = re.match(r'(\d+:\d+) - (\d+:\d+) : (.+)', line)
            if match:
                start_time, end_time, narration = match.groups()
                start_frame = time_to_frame(start_time, fps)
                stop_frame = time_to_frame(end_time, fps)
                metadata.append({
                    "narration_text": narration.strip(),
                    "start_frame": start_frame,
                    "stop_frame": stop_frame
                })

    # 最后一个sample
    if sample_id and metadata:
        results.append({
            "sample_id": sample_id,
            "task_goal": parse_task_goal(sample_id),
            "task_progress_metadata": metadata
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    process_file(
        input_path="timestamp.txt",
        output_path="BAAI_navi.json",
        fps=24
    )