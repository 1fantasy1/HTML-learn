from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 初始化噪声抑制管道
ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='damo/speech_frcrn_ans_cirm_16k'
)

# 对提供的音频文件进行噪声抑制
result = ans(
    'F:\下载\《北京东路的日子》青春永不毕业！_高清-1080P60-Compressed-with-FlexClip.wav',
    output_path='output.wav'
)

# 检查结果是否成功，并打印输出路径
if result:
    print("噪声抑制完成。输出保存到:", result['output_path'])
else:
    print("噪声抑制失败。")
