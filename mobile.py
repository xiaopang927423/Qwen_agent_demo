from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from openai import OpenAI
from qwen_vl_utils import smart_resize
from PIL import Image
from mobile_tool import MobileUse
from transformers import AutoProcessor
from modelscope import snapshot_download
from screen_shot import take_screenshot_and_save
from ppadb.client import Client as AdbClient

import json
import base64
import os
import time

# 获取client
def get_client():
    print('正在连接手机')
    client = AdbClient(host="127.0.0.1", port=5037)
    device = client.devices()[0]
    print('连接手机成功')
    return device


# 获取processor
def get_processor():

    model_path = snapshot_download(
        # 模型仓库地址
        repo_id='Qwen/Qwen2.5-VL-7B-Instruct',
        # 模型安装路径
        cache_dir='D:/Qwen_demo/test/mobile_demo/model')

    # 加载处理器
    processor = AutoProcessor.from_pretrained(model_path)
    return processor



# 图片编码
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def mobile_action(user_quest,screenshot,model_id, device, processor,step=None):

    # 操作历史可以按照以下方式组织：step x: [action]；step x+1: [action]
    user_query = f"{user_quest}\nTask progress (You have done the following operation on the current device):{step} "

    client = OpenAI(
        # 配置API连接，请替换成自己的API Key
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 图像信息处理.
    dummy_image = Image.open(screenshot)
    resized_height, resized_width  = smart_resize(dummy_image.height,
        dummy_image.width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=processor.image_processor.max_pixels,)
    mobile_use = MobileUse(
        cfg={"display_width_px": resized_width, "display_height_px": resized_height}, device=device
    )

    # 图片编码
    base64_image = encode_image(screenshot)

    # 构建message
    system_message = NousFnCallPrompt.preprocess_fncall_messages(
        messages=[
            Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
        ],
        functions=[mobile_use.function],
        lang=None,
    )

    system_message = system_message[0].model_dump()
    messages=[
        {
            "role": "system",
            "content": [
                {"type": "text", "text": msg["text"]} for msg in system_message["content"]
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "min_pixels": processor.image_processor.min_pixels,
                    "max_pixels": processor.image_processor.max_pixels,
                    # Pass in BASE64 image data. Note that the image format (i.e., image/{format}) must match the Content Type in the list of supported images. "f" is the method for string formatting.
                    # PNG image:  f"data:image/png;base64,{base64_image}"
                    # JPEG image: f"data:image/jpeg;base64,{base64_image}"
                    # WEBP image: f"data:image/webp;base64,{base64_image}"
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
                {"type": "text", "text": user_query},
            ],
        }
    ]
    completion = client.chat.completions.create(
        model = model_id,
        messages = messages,
    )

    #模型输出信息处理
    output_text = completion.choices[0].message.content
    print(completion)
    action = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
    if not action['arguments'] == 'terminate' :
        mobile_use.call(action['arguments'])
        time.sleep(5)
    return action

def mian(local_directory='D:/Qwen_demo/test/mobile_demo/assets/screenshot',screenshot_name='screenshot.png'):
    processor = get_processor()
    device = get_client()
    screen_shot = take_screenshot_and_save(device, local_directory, screenshot_name)
    step = 1
    steps = ''
    prompt = input('请描述你要完成的任务：')
    while True:
        if step > 15:
            break

        next_action = mobile_action(f"The user query:{prompt}", screen_shot, "qwen2.5-vl-7b-instruct", device, processor,steps)
        screen_shot = take_screenshot_and_save(device, local_directory, screenshot_name)
        if next_action['arguments']['action'] == 'terminate':
            if next_action['arguments']['status'] == 'success':
                print('操作成功')
                break
            else:
                print('操作失败')
                break
        str_step = str(next_action)
        print(next_action)
        str_step_num = str(step)
        steps += f'Step{str_step_num}: ' + str_step + ','
        step += 1

if __name__ == '__main__':
    mian()