"""

```bash
python3 inference/gradio_web_demo_custom.py -c /home/jiangzhu/project/models/t2i/CogView4-6B --mode 3 --server-port 7134
```
"""


# %% logging 初始化
import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)


# %%
import gc
import random
import re
import threading
import time
from argparse import ArgumentParser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, List

import gradio as gr
import torch
from diffusers import CogView4Pipeline
from diffusers.models import CogView4Transformer2DModel
from torchao.quantization import int8_weight_only, quantize_
from transformers import GlmModel


# %%
DEFAULT_CKPT_PATH = "model/CogView4-6B"
MAX_PIXELS = 2 ** 21  # 2097152，用于限制 width * height
FILE_LOCK = threading.Lock()  # 文件操作锁
logger = logging.getLogger(__name__)


# %%
def _get_args() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument('-c', '--checkpoint-path', type=str, default=DEFAULT_CKPT_PATH,
                        help='模型文件路径，默认 %(default)r')
    parser.add_argument('--mode', type=int, default=0,
                        help="显存/内存模式：12G→1；24G 32G→2；24G 64G→3；40G↑→0(默认)")
    parser.add_argument('--inbrowser', action='store_true', default=False,
                        help='启动后自动在浏览器打开')
    parser.add_argument('--server-port', type=int, default=7134, help='Demo server port.')
    parser.add_argument('--server-name', type=str, default='0.0.0.0', help='Demo server name.')
    args = parser.parse_args()
    return args


def _print_system_info() -> None:
    if not torch.cuda.is_available():
        logger.warning("未检测到 GPU，所有计算将在 CPU 上运行。")
        return
    total_vram_in_gb = torch.cuda.get_device_properties(0).total_memory / 1073741824
    logger.info(f"CUDA版本：{torch.version.cuda}")
    logger.info(f"Pytorch版本：{torch.__version__}")
    logger.info(f"显卡型号：{torch.cuda.get_device_name()}")
    logger.info(f"显存大小：{total_vram_in_gb:.2f}GB")


def _load_pipe(args) -> CogView4Pipeline:
    """
    根据显存模式加载 CogView4Pipeline。
    mode:
        0 —— 大显存，原生 fp16/bf16，尝试 xFormers
        1 —— 12 GB 显存，用 int8 量化 + CPU Offload
        2 —— 24 GB 显存 / 32 GB 内存，用 int8 量化（无 Offload）
        3 —— 24 GB 显存 / 64 GB 内存，用原生精度 + CPU Offload
    """
    # 参数合法性校验
    if args.mode not in [0, 1, 2, 3]:
        raise ValueError(f"Invalid mode: {args.mode}, must be integer 0-3")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    major, _ = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    logger.info(f"使用数据类型：{dtype}")

    # ---------------- 手动准备量化组件（仅 mode 1/2） ----------------
    extra_components = {}
    if args.mode in [1, 2]:
        logger.info("进入小显存模式，开始加载并量化子模块…")
        text_encoder = GlmModel.from_pretrained(
            args.checkpoint_path, subfolder="text_encoder", torch_dtype=dtype)
        transformer = CogView4Transformer2DModel.from_pretrained(
            args.checkpoint_path, subfolder="transformer", torch_dtype=dtype)
        quantize_(text_encoder, int8_weight_only())
        quantize_(transformer, int8_weight_only())
        extra_components.update(text_encoder=text_encoder, transformer=transformer)
        logger.info("已完成 int8 weight-only 量化")

    # ---------------- 创建 Pipeline ----------------
    pipe = CogView4Pipeline.from_pretrained(
        args.checkpoint_path,
        torch_dtype=dtype,
        **extra_components,          # 只有需要时才传入组件
    ).to(device)

    # 小显存模式 1/3 → CPU Offload
    if args.mode in [1, 3]:
        pipe.enable_model_cpu_offload()
        logger.info("已启用 CPU Offload")

    # VAE 分片 / 平铺，加速+省显存（任何模式都安全）
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


def clean_string(s: str) -> str:
    """去除多余空格/换行"""
    return re.sub(r"\s+", " ", s.replace("\n", " ").strip())


def delete_old_files() -> None:
    """定时清理 gradio_tmp 中文件"""
    tmp_dir = Path("./gradio_tmp")
    tmp_dir.mkdir(exist_ok=True)
    while True:
        try:
            with FILE_LOCK:  # 加文件锁
                cutoff = datetime.now() - timedelta(minutes=5)
                for f in tmp_dir.iterdir():
                    if f.is_file() and datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
                        try:
                            f.unlink()
                            logger.debug(f"已删除临时文件 {f}")
                        except Exception as e:
                            logger.error(f"删除文件失败 {f}: {str(e)}")
            time.sleep(600)
        except Exception as e:
            logger.error(f"文件清理线程异常: {str(e)}")
            time.sleep(60)


# %% ---------------------------- 推理主函数 ----------------------------
@torch.inference_mode()
def _infer(
    pipe: CogView4Pipeline,
    prompt: str,
    seed: int,
    randomize_seed: bool,
    width: int,
    height: int,
    guidance_scale: float,
    num_inference_steps: int,
    num_images: int,
) -> Tuple[List, int]:

    # 输入参数验证: 宽高在128-2048之间，且为32的倍数，图片数量在1-8之间
    width = max(128, min(width, 2048)) // 32 * 32
    height = max(128, min(height, 2048)) // 32 * 32
    num_images = int(max(1, min(num_images, 8)))  # 强制转换为整数

    if width * height > MAX_PIXELS:
        raise gr.Error(f"分辨率过大（{width}×{height}），请降低尺寸")

    if randomize_seed:
        seed = random.randint(0, 65536)

    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        prompt = clean_string(prompt)
        images = pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
        ).images

        return images

    except Exception as e:
        logger.error(f"推理失败: {str(e)}")
        raise gr.Error("生成失败，请调整参数后重试") from e
    finally:
        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# %% ---------------------------- Gradio 前端 ----------------------------
def _launch_demo(args, pipe: CogView4Pipeline) -> None:
    def update_max_height(width):
        return gr.update(maximum=MAX_PIXELS // max(width, 1))

    def update_max_width(height):
        return gr.update(maximum=MAX_PIXELS // max(height, 1))

    def infer_wrapped(*inputs):
        return _infer(pipe, *inputs)

    with gr.Blocks(theme=gr.themes.Soft()) as demo:

        examples = [
            "这是一幅充满皮克斯风格的动画渲染图像，展现了一只拟人化的粘土风格小蛇。这条快乐的小蛇身着魔术师装扮，占据了画面下方三分之一的位置，显得俏皮而生动。它的头上戴着一顶黑色羊毛材质的复古礼帽，身上穿着一件设计独特的红色棉袄，白色的毛袖增添了一抹温暖的对比。小蛇的鳞片上精心绘制了金色梅花花纹，显得既华丽又不失可爱。它的腹部和脸庞呈现洁白，与红色的身体形成鲜明对比。 这条蜿蜒的小蛇拥有可爱的塑胶手办质感，仿佛随时会从画面中跃然而出。背景是一片鲜艳的红色，地面上散布着宝箱、金蛋和红色灯笼等装饰物，营造出浓厚的节日气氛。画面的上半部分用金色连体字书写着 “Happy New Year”，庆祝新年的到来，同时也暗示了蛇年的到来，为整幅画面增添了一份节日的喜悦和祥瑞。",
            "在这幅如梦似幻的画作中，一辆由云朵构成的毛绒汽车轻盈地漂浮在蔚蓝的高空之中。这辆汽车设计独特，车身完全由洁白、蓬松的云朵编织而成，每一处都散发着柔软而毛茸茸的质感。从车顶到轮胎，再到它的圆润车灯，无一不是由细腻的云丝构成，仿佛随时都可能随风轻轻摆动。车窗也是由透明的云物质构成，同样覆盖着一层细软的绒毛，让人不禁想要触摸。 这辆神奇的云朵汽车仿佛是魔法世界中的交通工具，它悬浮在夕阳映照的绚丽天空之中，周围是五彩斑斓的晚霞和悠然飘浮的云彩。夕阳的余晖洒在云朵车上，为其柔软的轮廓镀上了一层金色的光辉，使得整个场景既温馨又神秘，引人入胜。",
            "A vintage red convertible with gleaming chrome finishes sits attractively under the golden hues of a setting sun, parked on a deserted cobblestone street in a charming old town. The car's polished body reflects the surrounding quaint buildings and the few early evening stars beginning to twinkle in the gentle gradient of the twilight sky. A light breeze teases the few fallen leaves near the car's pristine white-walled tires, which rest casually by the sidewalk, hinting at the leisurely pace of life in this serene setting.",
        ]

        gr.Markdown("""
                <div>
                    <h2 style="font-size: 30px;text-align: center;">中关村银行-文生图体验</h2>
                    <h3 style="text-align: center;">使用模型:【CogView4-6B】</h3>
                </div>
            """)

        with gr.Row():
            with gr.Column():
                # ------ 主输入区 ------
                prompt = gr.Text(label="提示词", show_label=False, max_lines=15, lines=5,
                                 placeholder="描述要生成的图像，可先用大语言模型优化后再输入")
                run_button = gr.Button("运行", variant="primary")

                # ------ 折叠参数区 ------
                with gr.Accordion(label="参数设置", open=False):
                    with gr.Row():
                        with gr.Column():
                            num_images = gr.Number(label="生成数量", value=1, minimum=1, maximum=8, step=1, precision=0)
                            width = gr.Slider(128, 2048, 512, 32, label="宽度(px)")
                            height = gr.Slider(128, 2048, 512, 32, label="高度(px)")
                        with gr.Column():
                            with gr.Row():
                                seed = gr.Slider(label="种子（控制随机性）", minimum=0, maximum=65536, step=1, value=0, scale=3)
                                randomize_seed = gr.Checkbox(label="随机种子", value=True, scale=1)
                            guidance_scale = gr.Slider(0.0, 10.0, 5.0, 0.1, label="提示词影响程度")
                            num_inference_steps = gr.Slider(10, 100, 50, 1, label="推理步数")
            with gr.Column():
                # 生成多张图片
                result = gr.Gallery(label="生成结果", format="png")
                clear = gr.ClearButton(components=[prompt, result], value="清除内容")

        # ➜ 控件联动
        width.change(update_max_height, inputs=[width], outputs=[height])
        height.change(update_max_width, inputs=[height], outputs=[width])

        with gr.Row():
            # ➜ 示例
            examples = [
                ["这是一幅充满皮克斯风格的动画渲染图像，展现了一只拟人化的粘土风格小蛇。这条快乐的小蛇身着魔术师装扮，占据了画面下方三分之一的位置，显得俏皮而生动。它的头上戴着一顶黑色羊毛材质的复古礼帽，身上穿着一件设计独特的红色棉袄，白色的毛袖增添了一抹温暖的对比。小蛇的鳞片上精心绘制了金色梅花花纹，显得既华丽又不失可爱。它的腹部和脸庞呈现洁白，与红色的身体形成鲜明对比。 这条蜿蜒的小蛇拥有可爱的塑胶手办质感，仿佛随时会从画面中跃然而出。背景是一片鲜艳的红色，地面上散布着宝箱、金蛋和红色灯笼等装饰物，营造出浓厚的节日气氛。画面的上半部分用金色连体字书写着 “Happy New Year”，庆祝新年的到来，同时也暗示了蛇年的到来，为整幅画面增添了一份节日的喜悦和祥瑞。"],
                ["在这幅如梦似幻的画作中，一辆由云朵构成的毛绒汽车轻盈地漂浮在蔚蓝的高空之中。这辆汽车设计独特，车身完全由洁白、蓬松的云朵编织而成，每一处都散发着柔软而毛茸茸的质感。从车顶到轮胎，再到它的圆润车灯，无一不是由细腻的云丝构成，仿佛随时都可能随风轻轻摆动。车窗也是由透明的云物质构成，同样覆盖着一层细软的绒毛，让人不禁想要触摸。 这辆神奇的云朵汽车仿佛是魔法世界中的交通工具，它悬浮在夕阳映照的绚丽天空之中，周围是五彩斑斓的晚霞和悠然飘浮的云彩。夕阳的余晖洒在云朵车上，为其柔软的轮廓镀上了一层金色的光辉，使得整个场景既温馨又神秘，引人入胜。"],
                ["A vintage red convertible with gleaming chrome finishes sits attractively under the golden hues of a setting sun, parked on a deserted cobblestone street in a charming old town. The car's polished body reflects the surrounding quaint buildings and the few early evening stars beginning to twinkle in the gentle gradient of the twilight sky. A light breeze teases the few fallen leaves near the car's pristine white-walled tires, which rest casually by the sidewalk, hinting at the leisurely pace of life in this serene setting."],
            ]
            gr.Examples(examples=examples, inputs=prompt,
                        label="提示词示例（点击即可填入上方文本框）", cache_examples=False)

        gr.on(
            triggers=[run_button.click, prompt.submit], fn=infer_wrapped,
            inputs=[prompt, seed, randomize_seed, width, height,
                    guidance_scale, num_inference_steps, num_images],
            outputs=[result],
        )

    demo.queue().launch(
        inbrowser=args.inbrowser,
        server_name=args.server_name,
        server_port=args.server_port,
    )


def main():
    _print_system_info()

    # 背景线程：临时文件清理
    threading.Thread(target=delete_old_files, daemon=True).start()

    args = _get_args()
    pipe = _load_pipe(args)
    _launch_demo(args, pipe)


if __name__ == "__main__":
    main()