# 导入必要的库
from modelscope.pipelines import pipeline as pipeline_ali
from modelscope.utils.constant import Tasks
from moviepy.editor import VideoFileClip
import os
import ffmpeg
from faster_whisper import WhisperModel
import math
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# 指定本地模型存储目录
local_dir_root = "./models_from_modelscope"

# 定义模型路径
model_dir_cirm = './models_from_modelscope/damo/speech_frcrn_ans_cirm_16k'
model_dir_ins = './models_from_modelscope/damo/nlp_csanmt_translation_en2zh'

# 定义设备类型
device = "cuda" if torch.cuda.is_available() else "cpu"


# 定义合并字幕的函数
def merge_sub(video_path, srt_path):
    # 删除已存在的输出文件
    if os.path.exists("./test_srt.mp4"):
        os.remove("./test_srt.mp4")

    # 使用ffmpeg合并视频和字幕
    ffmpeg.input(video_path).output("./test_srt.mp4", vf="subtitles=" + srt_path).run()

    return "./test_srt.mp4"


# 定义翻译字幕的函数，日语到中文
def make_tran_ja2zh_neverLife(srt_path):
    # 加载模型和分词器
    model_path = "neverLife/nllb-200-distilled-600M-ja-zh"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, from_pt=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang="jpn_Jpan", tgt_lang="zho_Hans", from_pt=True)

    # 读取字幕文件
    with open(srt_path, 'r', encoding="utf-8") as file:
        gweight_data = file.read()

    # 分割字幕文件内容
    result = gweight_data.split("\n\n")

    # 删除已存在的输出文件
    if os.path.exists("./two.srt"):
        os.remove("./two.srt")

    # 遍历字幕文件内容并翻译
    for res in result:
        line_srt = res.split("\n")
        try:
            input_ids = tokenizer.encode(line_srt[2], max_length=128, padding=True, return_tensors='pt')
            outputs = model.generate(input_ids, num_beams=4, max_new_tokens=128)
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(translated_text)
        except IndexError as e:
            print(f"翻译完毕")
            break
        except Exception as e:
            print(str(e))

        # 将翻译结果写入文件
        with open("./two.srt", "a", encoding="utf-8") as f:
            f.write(f"{line_srt[0]}\n{line_srt[1]}\n{line_srt[2]}\n{translated_text}\n\n")

    # 读取翻译后的字幕文件
    with open("./two.srt", "r", encoding="utf-8") as f:
        content = f.read()

    return content


# 定义翻译字幕的函数，韩语到中文
def make_tran_ko2zh(srt_path):
    # 加载模型和分词器
    model_path = "./model_from_hg/ko-zh/"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

    # 读取字幕文件
    with open(srt_path, 'r', encoding="utf-8") as file:
        gweight_data = file.read()

    # 分割字幕文件内容
    result = gweight_data.split("\n\n")

    # 删除已存在的输出文件
    if os.path.exists("./two.srt"):
        os.remove("./two.srt")

    # 遍历字幕文件内容并翻译
    for res in result:
        line_srt = res.split("\n")
        try:
            input_ids = tokenizer.encode(line_srt[2], max_length=128, padding=True, return_tensors='pt')
            outputs = model.generate(input_ids, num_beams=4, max_new_tokens=128)
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(translated_text)
        except IndexError as e:
            print(f"翻译完毕")
            break
        except Exception as e:
            print(str(e))

        # 将翻译结果写入文件
        with open("./two.srt", "a", encoding="utf-8") as f:
            f.write(f"{line_srt[0]}\n{line_srt[1]}\n{line_srt[2]}\n{translated_text}\n\n")

    # 读取翻译后的字幕文件
    with open("./two.srt", "r", encoding="utf-8") as f:
        content = f.read()

    return content


# 定义翻译字幕的函数，日语到中文
def make_tran_ja2zh(srt_path):
    # 加载模型和分词器
    model_path = "./model_from_hg/ja-zh/"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

    # 读取字幕文件
    with open(srt_path, 'r', encoding="utf-8") as file:
        gweight_data = file.read()

    # 分割字幕文件内容
    result = gweight_data.split("\n\n")

    # 删除已存在的输出文件
    if os.path.exists("./two.srt"):
        os.remove("./two.srt")

    # 遍历字幕文件内容并翻译
    for res in result:
        line_srt = res.split("\n")
        try:
            input_ids = tokenizer.encode(f'<-ja2zh-> {line_srt[2]}', max_length=128, padding=True, return_tensors='pt')
            outputs = model.generate(input_ids, num_beams=4, max_new_tokens=128)
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(translated_text)
        except IndexError as e:
            print(f"翻译完毕")
            break
        except Exception as e:
            print(str(e))

        # 将翻译结果写入文件
        with open("./two.srt", "a", encoding="utf-8") as f:
            f.write(f"{line_srt[0]}\n{line_srt[1]}\n{line_srt[2]}\n{translated_text}\n\n")

    # 读取翻译后的字幕文件
    with open("./two.srt", "r", encoding="utf-8") as f:
        content = f.read()

    return content


# 定义翻译字幕的函数，中文到英文
def make_tran_zh2en(srt_path):
    # 加载模型和分词器
    model_path = "./model_from_hg/zh-en/"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

    # 读取字幕文件
    with open(srt_path, 'r', encoding="utf-8") as file:
        gweight_data = file.read()

    # 分割字幕文件内容
    result = gweight_data.split("\n\n")

    # 删除已存在的输出文件
    if os.path.exists("./two.srt"):
        os.remove("./two.srt")

    # 遍历字幕文件内容并翻译
    for res in result:
        line_srt = res.split("\n")
        try:
            tokenized_text = tokenizer.prepare_seq2seq_batch([line_srt[2]], return_tensors='pt')
            translation = model.generate(**tokenized_text)
            translated_text = tokenizer.batch_decode(translation, skip_special_tokens=False)[0]
            translated_text = translated_text.replace("<pad>", "").replace("</s>", "").strip()
            print(translated_text)
        except IndexError as e:
            print(f"翻译完毕")
            break
        except Exception as e:
            print(str(e))

        # 将翻译结果写入文件
        with open("./two.srt", "a", encoding="utf-8") as f:
            f.write(f"{line_srt[0]}\n{line_srt[1]}\n{line_srt[2]}\n{translated_text}\n\n")

    # 读取翻译后的字幕文件
    with open("./two.srt", "r", encoding="utf-8") as f:
        content = f.read()

    return content


# 定义翻译字幕的函数，英文到中文
def make_tran(srt_path):
    # 加载模型和分词器
    model_path = "./model_from_hg/en-zh/"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

    # 读取字幕文件
    with open(srt_path, 'r', encoding="utf-8") as file:
        gweight_data = file.read()

    # 分割字幕文件内容
    result = gweight_data.split("\n\n")

    # 删除已存在的输出文件
    if os.path.exists("./two.srt"):
        os.remove("./two.srt")

    # 遍历字幕文件内容并翻译
    for res in result:
        line_srt = res.split("\n")
        try:
            tokenized_text = tokenizer.prepare_seq2seq_batch([line_srt[2]], return_tensors='pt')
            translation = model.generate(**tokenized_text)
            translated_text = tokenizer.batch_decode(translation, skip_special_tokens=False)[0]
            translated_text = translated_text.replace("<pad>", "").replace("</s>", "").strip()
            print(translated_text)
        except IndexError as e:
            print(f"翻译完毕")
            break
        except Exception as e:
            print(str(e))

        # 将翻译结果写入文件
        with open("./two.srt", "a", encoding="utf-8") as f:
            f.write(f"{line_srt[0]}\n{line_srt[1]}\n{line_srt[2]}\n{translated_text}\n\n")

    # 读取翻译后的字幕文件
    with open("./two.srt", "r", encoding="utf-8") as f:
        content = f.read()

    return content


# 定义提取视频人声的函数
def movie2audio(video_path):
    # 读取视频文件
    video = VideoFileClip(video_path)

    # 提取视频文件中的声音
    audio = video.audio

    # 将声音保存为WAV格式
    audio.write_audiofile("./audio.wav")

    # 使用模型提取人声
    ans = pipeline_ali(
        Tasks.acoustic_noise_suppression,
        model=model_dir_cirm)

    ans('./audio.wav', output_path='./output.wav')

    return "./output.wav"


# 定义制作字幕文件的函数
def make_srt(file_path, model_name="small"):
    # 判断设备类型并加载模型
    if device == "cuda":
        try:
            model = WhisperModel(model_name, device="cuda", compute_type="float16",
                                 download_root="./model_from_whisper", local_files_only=False)
        except Exception as e:
            model = WhisperModel(model_name, device="cuda", compute_type="int8_float16",
                                 download_root="./model_from_whisper", local_files_only=False)
    else:
        model = WhisperModel(model_name, device="cpu", compute_type="int8", download_root="./model_from_whisper",
                             local_files_only=False)

    # 识别视频中的语音并生成字幕
    segments, info = model.transcribe(file_path, beam_size=5, vad_filter=True,
                                      vad_parameters=dict(min_silence_duration_ms=500))

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    count = 0
    with open('./video.srt', 'w', encoding="utf-8") as f:  # Open file for writing
        for segment in segments:
            count += 1
            duration = f"{convert_seconds_to_hms(segment.start)} --> {convert_seconds_to_hms(segment.end)}\n"
            text = f"{segment.text.lstrip()}\n\n"

            f.write(f"{count}\n{duration}{text}")  # Write formatted string to the file
            print(f"{duration}{text}", end='')

    with open("./video.srt", "r", encoding="utf-8") as f:
        content = f.read()

    return content


# 定义将秒转换为时分秒的函数
def convert_seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = math.floor((seconds % 1) * 1000)
    output = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"
    return output


# 制作字幕文件
def make_srt(file_path, model_name="small"):
    """
    使用Whisper模型从视频文件中生成字幕。
    参数:
    file_path: 视频文件路径
    model_name: Whisper模型的名称，默认为"small"
    返回:
    字幕文件的内容
    """
    # 根据设备类型加载Whisper模型
    if device == "cuda":
        try:
            model = WhisperModel(model_name, device="cuda", compute_type="float16",
                                 download_root="./model_from_whisper", local_files_only=False)
        except Exception as e:
            model = WhisperModel(model_name, device="cuda", compute_type="int8_float16",
                                 download_root="./model_from_whisper", local_files_only=False)
    else:
        model = WhisperModel(model_name, device="cpu", compute_type="int8", download_root="./model_from_whisper",
                             local_files_only=False)

    # 从视频文件中转录语音，并获取相关信息
    segments, info = model.transcribe(file_path, beam_size=5, vad_filter=True,
                                      vad_parameters=dict(min_silence_duration_ms=500))

    # 打印检测到的语言及其概率
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    count = 0
    # 打开文件用于写入字幕
    with open('./video.srt', 'w', encoding="utf-8") as f:
        # 遍历转录结果，生成字幕
        for segment in segments:
            count += 1
            # 将时间戳转换为时分秒格式
            duration = f"{convert_seconds_to_hms(segment.start)} --> {convert_seconds_to_hms(segment.end)}\n"
            # 获取转录文本
            text = f"{segment.text.lstrip()}\n\n"
            # 将字幕写入文件
            f.write(f"{count}\n{duration}{text}")
            # 打印字幕内容
            print(f"{duration}{text}", end='')

    # 读取生成的字幕文件内容
    with open("./video.srt", "r", encoding="utf-8") as f:
        content = f.read()

    return content


# 提取人声
def movie2audio(video_path):
    """
    从视频文件中提取人声。
    参数:
    video_path: 视频文件路径
    返回:
    提取的人声文件路径
    """
    # 读取视频文件
    video = VideoFileClip(video_path)

    # 提取视频文件中的声音
    audio = video.audio

    # 将声音保存为WAV格式
    audio.write_audiofile("./audio.wav")

    # 使用噪声抑制模型处理声音文件
    ans = pipeline_ali(
        Tasks.acoustic_noise_suppression,
        model=model_dir_cirm)

    # 处理声音文件并保存输出
    ans('./audio.wav', output_path='./output.wav')

    return "./output.wav"