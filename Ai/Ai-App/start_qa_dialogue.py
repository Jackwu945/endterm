import io
import re
import json
import time
import wave
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM


# import simpleaudio as sa

# from mlx_lm import load, generate

# def generate_speech(text, port):
#     time_ckpt = time.time()
#     data={
#         "text": text,
#         "text_language": "zh"
#     }
#     response = requests.post("http://127.0.0.1:{}".format(port), json=data)
#     if response.status_code == 400:
#         raise Exception(f"GPT-SoVITS ERROR: {response.message}")
#     audio_data = io.BytesIO(response.content)
#     with wave.open(audio_data, 'rb') as wave_read:
#         audio_frames = wave_read.readframes(wave_read.getnframes())
#         audio_wave_obj = sa.WaveObject(audio_frames, wave_read.getnchannels(), wave_read.getsampwidth(), wave_read.getframerate())
#     play_obj = audio_wave_obj.play()
#     play_obj.wait_done()
#     print("Audio Generation Time: %d ms\n" % ((time.time() - time_ckpt) * 1000))

# 定义 generate 方法
def generate(model, tokenizer, prompt, temp=0.3, max_tokens=500, verbose=False):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_tokens,
        temperature=temp,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def split_text(text):
    sentence_endings = ['！', '。', '？'," "]
    for punctuation in sentence_endings:
        text = text.replace(punctuation, punctuation + '\n')
    pattern = r'\[.*?\]'
    text = re.sub(pattern, '', text)
    return text

if __name__ == "__main__":
    # with open('../models/Qwen1.5-1.8B-Chat-token/tokenizer_config.json', 'r') as file:
    #     tokenizer_config = json.load(file)

    with open('../models/merged/tokenizer_config.json', 'r') as file:
        tokenizer_config = json.load(file)

        # # 加载模型
        # model_path = "../models/Qwen1.5-1.8B-Chat/"  # 运行前请修改为本地的模型路径
        # token_path = "../models/Qwen1.5-1.8B-Chat-token/"
        # 加载模型
        model_path = "../models/merged/"  # 运行前请修改为本地的模型路径
        token_path = "../models/merged/"
        tokenizer = AutoTokenizer.from_pretrained(token_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

    questions = []
    with open('../text/questions.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                questions.append(line)

    sys_msg = 'You are a annoying assistant'

    with open('../text/chat_template.txt', 'r',encoding='utf-8') as template_file:
        template = template_file.read()

    for question in questions:
        prompt = template.replace("{usr_msg}", question)
        print("%s" % (question))
        question = split_text(question)
        # generate_speech(question, 9880)
        
        time_ckpt = time.time()
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            temp=0.3,
            max_tokens=300,
            verbose=False
        )

        print("%s (Time %d ms)\n" % (response, (time.time() - time_ckpt) * 1000))
        response = split_text(response)
        # generate_speech(response, 9881)