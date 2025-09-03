from openai import AzureOpenAI  # 导入Azure专用客户端
import base64
import json
import time
import threading
import concurrent.futures
import os
from queue import Queue

system_prompt='''
I am a robot that cannot go through walls and must use doors or open gates to navigate across different areas. 
This is the floor plan of the building I am in right now (provided as an image). 
The pathwars are corridors are green areas. The doors and open gates which exist are represented as dark blue rectangles and labeled with 'door' and 'open'. The black lines are all walls. 
Alongside the image, make a clear list of all areas, doors, open gates with their connections - which is to be used for the navigation task.
'''

prompt = '''    
My task now is: {goal}. Please make a step by step solution for the task.
'''


# 配置 Azure OpenAI API
# openai.api_type = "azure"
# openai.api_key = os.getenv("OPENAI_API_KEY"),  # 从环境变量读取API密钥
# openai.api_base = "https://yjzl-openai-zywang.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview"
# openai.api_version = "2025-01-01"  # 替换为你所使用的 API 版本


# 初始化Azure客户端
client = AzureOpenAI(
    azure_endpoint="https://yjzl-openai-zywang.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview",  # Azure终结点
    api_key=os.getenv("OPENAI_API_KEY"),  # 从环境变量读取API密钥,  # API密钥
    api_version="2025-04-01-preview"  # API版本
)

# Azure 模型部署名称
deployment_name = "gpt-4o-2024-05-13"

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_messages(image_paths, texts):
    messages = [{"role": "system", "content": "I am a robot that cannot go through walls and must use opens to navigate across different areas. This is the floor plan of the building I am in right now (provided as an image). "}]
    for path, text in zip(image_paths, texts):
        image_b64 = encode_image(path)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        })
    return messages

def query_gpt4o_azure(messages, deployment_name):
    response = client.chat.completions.create(  # 新版接口
        model=deployment_name,  # 部署名称赋值给model参数
        messages=messages,
        max_tokens=4096,
        temperature=0.7
    )
    return response.choices[0].message.content  # 对象属性访问（非字典）


def format_data_sft_with_step(sample):
    formatted_sample = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a Vision Language Model specialized in processing the first-person-view images of embodied robots. Your task is to analyze the provided image and respond to queries with answers. Focus on the spatial relations in the image observations and make the right decisions."}],
        }
    ]
    tmp_content = []
    pos = 0
    for image, action in sample['image_action_pairs']:
        image_b64 = encode_image(image)
        if pos == 0:
            tmp_content += [
                {
                    "type": "text",
                    "text": sample["query"],
                },
                {
                    "type": "text",
                    "text": f"Observation #{str(pos + 1)}:\n",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {
                    "type": "text",
                    "text": f"Action #{str(pos + 1)}:\n{action}\n",
                },
            ]
            # formatted_sample += [ 
            #     {
            #         "role": "user",
            #         "content": [
            #             {
            #                 "type": "text",
            #                 "text": sample["query"],
            #             },
            #             {
            #                 "type": "text",
            #                 "text": f"Observation #{str(pos + 1)}:\n",
            #             },
            #             {
            #                 "type": "image_url",
            #                 "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            #             },
            #             {
            #                 "type": "text",
            #                 "text": f"Action #{str(pos + 1)}:\n{action}",
            #             },
            #         ],
            #     }
            # ]
        else:
            tmp_content += [
                {
                    "type": "text",
                    "text": f"Observation #{str(pos + 1)}:\n",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {
                    "type": "text",
                    "text": f"Action #{str(pos + 1)}:\n{action}\n",
                },
            ]
        pos += 1

    tmp_content += [
        {
            "type": "text",
            "text": prompt_query_text,
        }, 
    ] 

    
    formatted_sample += [
        {
            "role": "user",
            "content": tmp_content,
        }, 
    ]


    return formatted_sample

# # 示例调用
# if __name__ == "__main__":
#     file = "./text_action_data/filtered_data_v2.json"
#     with open(file, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     for each_data in data:
#         messages = format_data_sft_with_step(each_data)
#         print("messages = ", messages)
#         result = query_gpt4o_azure(messages, deployment_name)
#         print(result)
#         exit()



# 全局锁用于安全写入
file_lock = threading.Lock()

def process_data(data, progress_file, temp_output_file, result_queue):
    """处理单个数据样本的线程函数"""
    try:
        # 1. 格式化和查询
        messages = format_data_sft_with_step(data)
        result = query_gpt4o_azure(messages, deployment_name)
        
        # 2. 添加结果到数据结构
        data["reason_output"] = result
        
        # 3. 放入结果队列
        result_queue.put(data)
        
    except Exception as e:
        print(f"处理失败: {str(e)}")
        data["reason_output"] = f"ERROR: {str(e)}"
        result_queue.put(data)

def save_results(result_queue, progress_file, temp_output_file, total_count):
    """保存结果的线程函数"""
    processed_count = 0
    while processed_count < total_count:
        data = result_queue.get()  # 阻塞获取结果
        if data is None:  # 结束信号
            break
            
        # 安全写入临时文件
        with file_lock:
            # 读取现有数据
            if os.path.exists(temp_output_file):
                with open(temp_output_file, 'r', encoding='utf-8') as f:
                    all_data = json.load(f)
            else:
                all_data = []
                
            # 更新数据
            all_data.append(data)
            
            # 写入更新后的数据
            with open(temp_output_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, indent=2)
                
            # 更新进度
            processed_count += 1
            with open(progress_file, 'w') as pf:
                pf.write(str(processed_count))
                
            print(f"已保存进度: {processed_count}/{total_count}")

def main_processing():
    # 文件路径配置
    input_file = "./text_action_data/filtered_data_v2.json"
    progress_file = "./progress.txt"
    temp_output_file = "./temp_results.json"
    final_output_file = "./final_results.json"
    
    # 1. 加载数据并初始化进度
    with open(input_file, "r", encoding="utf-8") as f:
        full_data = json.load(f)
    
    total_count = len(full_data)
    start_index = 0
    
    # 2. 断点续传检查
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as pf:
            start_index = int(pf.read().strip())
            print(f"检测到进度，从索引 {start_index} 继续")
    
    # 3. 创建结果队列和工作线程
    result_queue = Queue()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # 启动保存线程
        saver_thread = threading.Thread(
            target=save_results,
            args=(result_queue, progress_file, temp_output_file, total_count)
        )
        saver_thread.start()
        
        # 4. 提交任务到线程池
        for i in range(start_index, total_count):
            # 添加延迟控制请求频率（每0.5秒一个新请求）
            time.sleep(0.5)
            executor.submit(process_data, full_data[i], progress_file, temp_output_file, result_queue)
        
        # 5. 等待所有任务完成
        executor.shutdown(wait=True)
        result_queue.put(None)  # 发送结束信号
        saver_thread.join()
    
    # 6. 最终处理
    # 重命名临时文件为最终输出
    if os.path.exists(temp_output_file):
        os.rename(temp_output_file, final_output_file)
    os.remove(progress_file)
    print(f"处理完成! 结果已保存到 {final_output_file}")

def single_query(img_path):
    messages = [{"role": "system", "content": system_prompt}]
    image_b64 = encode_image(img_path)
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt.format(goal="Go to robot training ground 1 and robot training ground 3 from kitchen")},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]
    })
    result = query_gpt4o_azure(messages, deployment_name)
    print(result)
    
# 运行主处理流程
if __name__ == "__main__":
    # main_processing()
    # with open("query_prompt_GT.txt", "r", encoding="utf-8") as f:
    #     prompt_query_text = f.read()
    single_query(img_path="./BAAI_semantic_floor_map_pathway.png")