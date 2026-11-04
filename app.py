
import io
import os
import time
import base64
import traceback
import threading

import torch
import cv2
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from PIL import Image, ImageDraw, ImageFont

# YOLOv5 7.0 原生推理模块
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes


# ==================================================
# 1. Flask 初始化
# ==================================================

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


# ==================================================
# 2. 参数配置
# ==================================================

MODEL_PATH = 'runs/train/exp13/weights/best_fixed.pt'

SHARED_DIR = 'shared_results'

MAX_SAVED_IMAGES = 30

CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

IMAGE_SIZE = 640

MAX_DETECTIONS = 1000


# ==================================================
# 3. 设备配置
# ==================================================

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

print('=' * 50)
print('道路病害智能检测系统')
print('=' * 50)

print('当前推理设备:', device)

if device.type == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0))


# ==================================================
# 4. 加载 YOLOv5 模型
# ==================================================

print('\n正在加载 YOLOv5 模型...')

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f'模型权重不存在: {MODEL_PATH}'
    )

# 使用 YOLOv5 原生 attempt_load
# 不使用 torch.hub.load
# 不使用 AutoShape
model = attempt_load(
    MODEL_PATH,
    device=device,
    inplace=True,
    fuse=True
)

model.to(device)
model.eval()

# 获取模型步长
STRIDE = int(model.stride.max())

# 获取类别名称
NAMES = model.module.names if hasattr(
    model, 'module'
) else model.names

print('模型加载成功！')
print('模型类型:', type(model))
print('模型步长:', STRIDE)
print('类别名称:', NAMES)
print('输入尺寸:', IMAGE_SIZE)

# 推理锁
inference_lock = threading.Lock()


# ==================================================
# 5. 文件夹初始化
# ==================================================

os.makedirs(
    SHARED_DIR,
    exist_ok=True
)


# ==================================================
# 6. 字体加载
# ==================================================

try:

    FONT_PATH = 'C:/Windows/Fonts/msyh.ttc'

    font = ImageFont.truetype(
        FONT_PATH,
        size=24
    )

except Exception:

    font = ImageFont.load_default()


# ==================================================
# 7. 图片解码
# ==================================================

def decode_image(image_data):

    if not image_data:

        raise ValueError('图片数据为空')

    if not isinstance(image_data, str):

        raise ValueError('图片数据必须是 Base64 字符串')

    # 兼容 Data URL
    if ',' in image_data:

        image_data = image_data.split(',', 1)[1]

    try:

        img_bytes = base64.b64decode(
            image_data,
            validate=True
        )

    except Exception:

        raise ValueError('Base64 图片数据格式错误')

    nparr = np.frombuffer(
        img_bytes,
        dtype=np.uint8
    )

    img = cv2.imdecode(
        nparr,
        cv2.IMREAD_COLOR
    )

    if img is None:

        raise ValueError('图片解码失败')

    return img


# ==================================================
# 8. YOLOv5 原生推理
# ==================================================

def run_inference(img):

    with inference_lock:

        with torch.inference_mode():

            # 原始图片尺寸
            original_height, original_width = img.shape[:2]

            # 1. Letterbox 缩放和填充
            img_letterbox = letterbox(
                img,
                IMAGE_SIZE,
                stride=STRIDE,
                auto=True
            )[0]

            # 2. BGR 转 RGB
            # HWC 转 CHW
            img_letterbox = img_letterbox[:, :, ::-1].transpose(
                2, 0, 1
            )

            img_letterbox = np.ascontiguousarray(
                img_letterbox
            )

            # 3. 转换为 Tensor
            img_tensor = torch.from_numpy(
                img_letterbox
            ).to(device)

            img_tensor = img_tensor.float()

            # 4. 归一化
            img_tensor /= 255.0

            # 5. 添加 Batch 维度
            if img_tensor.ndimension() == 3:

                img_tensor = img_tensor.unsqueeze(0)

            # 6. 模型推理
            # 注意：这里不再传递 size 参数
            predictions = model(
                img_tensor,
                augment=False
            )[0]

            # 7. NMS 非极大值抑制
            predictions = non_max_suppression(
                predictions,
                CONF_THRESHOLD,
                IOU_THRESHOLD,
                classes=None,
                agnostic=False,
                max_det=MAX_DETECTIONS
            )

            detections = []

            # 8. 处理检测结果
            for det in predictions:

                if len(det):

                    # 将检测框映射回原始图片尺寸
                    det[:, :4] = scale_boxes(
                        img_tensor.shape[2:],
                        det[:, :4],
                        img.shape
                    ).round()

                    for *xyxy, conf, cls in det:

                        x1, y1, x2, y2 = [
                            float(v.item()) for v in xyxy
                        ]

                        confidence = float(conf.item())

                        class_id = int(cls.item())

                        # 获取类别名称
                        if isinstance(NAMES, dict):

                            class_name = str(
                                NAMES[class_id]
                            )

                        else:

                            class_name = str(
                                NAMES[class_id]
                            )

                        # 坐标限制
                        x1 = max(
                            0,
                            min(x1, original_width - 1)
                        )

                        y1 = max(
                            0,
                            min(y1, original_height - 1)
                        )

                        x2 = max(
                            0,
                            min(x2, original_width - 1)
                        )

                        y2 = max(
                            0,
                            min(y2, original_height - 1)
                        )

                        detections.append({

                            'xmin': x1,
                            'ymin': y1,
                            'xmax': x2,
                            'ymax': y2,

                            'confidence': confidence,

                            'class_name': class_name

                        })

    # 返回 DataFrame，兼容原来的绘图函数
    return pd.DataFrame(
        detections,
        columns=[
            'xmin',
            'ymin',
            'xmax',
            'ymax',
            'confidence',
            'class_name'
        ]
    )


# ==================================================
# 9. 检测框绘制
# ==================================================

def draw_detections(img, df):

    img_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    pil_img = Image.fromarray(img_rgb)

    draw = ImageDraw.Draw(pil_img)

    width, height = pil_img.size

    detections = []

    for _, row in df.iterrows():

        x1 = max(
            0,
            min(int(row['xmin']), width - 1)
        )

        y1 = max(
            0,
            min(int(row['ymin']), height - 1)
        )

        x2 = max(
            0,
            min(int(row['xmax']), width - 1)
        )

        y2 = max(
            0,
            min(int(row['ymax']), height - 1)
        )

        cls_name = str(row['class_name'])

        conf = float(row['confidence'])

        detections.append({

            'xmin': float(x1),
            'ymin': float(y1),

            'xmax': float(x2),
            'ymax': float(y2),

            'confidence': conf,

            'class_name': cls_name

        })

        # 检测框
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(0, 255, 0),
            width=4
        )

        label = f'{cls_name} {conf:.2f}'

        bbox = draw.textbbox(
            (0, 0),
            label,
            font=font
        )

        text_width = bbox[2] - bbox[0]

        text_height = bbox[3] - bbox[1]

        text_y = y1 - text_height - 10

        if text_y < 0:

            text_y = y1

        text_x = min(
            max(0, x1),
            max(0, width - text_width - 8)
        )

        # 标签背景
        draw.rectangle(
            [
                text_x,
                text_y,

                min(
                    width - 1,
                    text_x + text_width + 8
                ),

                min(
                    height - 1,
                    text_y + text_height + 8
                )
            ],
            fill=(0, 255, 0)
        )

        # 标签文字
        draw.text(
            (text_x + 4, text_y + 4),
            label,
            font=font,
            fill=(0, 0, 0)
        )

    return pil_img, detections


# ==================================================
# 10. 图片转 Base64
# ==================================================

def image_to_base64(pil_img):

    buffered = io.BytesIO()

    pil_img.save(
        buffered,
        format='JPEG',
        quality=90
    )

    encoded = base64.b64encode(
        buffered.getvalue()
    ).decode('utf-8')

    return 'data:image/jpeg;base64,' + encoded


# ==================================================
# 11. 保存图片
# ==================================================

def save_pil_image(pil_img, prefix='result'):

    timestamp = time.time_ns() // 1_000_000

    filename = f'{prefix}_{timestamp}.jpg'

    save_path = os.path.join(
        SHARED_DIR,
        filename
    )

    pil_img.save(
        save_path,
        format='JPEG',
        quality=90
    )

    return save_path


# ==================================================
# 12. 历史图片清理
# ==================================================

def clean_old_images():

    files = []

    for filename in os.listdir(SHARED_DIR):

        if (
            filename.startswith('result_')
            and filename.endswith('.jpg')
        ):

            filepath = os.path.join(
                SHARED_DIR,
                filename
            )

            if os.path.isfile(filepath):

                files.append(filepath)

    files.sort(
        key=os.path.getmtime
    )

    if len(files) > MAX_SAVED_IMAGES:

        for old_file in files[:-MAX_SAVED_IMAGES]:

            try:

                os.remove(old_file)

                print('已删除旧图片:', old_file)

            except OSError as e:

                print('删除旧图片失败:', e)


# ==================================================
# 13. 首页
# ==================================================

@app.route('/')
def index():

    return render_template('index.html')


# ==================================================
# 14. 图片检测接口
# ==================================================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        data = request.get_json(
            silent=True
        )

        if not isinstance(data, dict):

            return jsonify({

                'status': 'error',

                'message': '请求数据格式错误'

            }), 400

        image_data = data.get('image')

        save_image = data.get(
            'save_image',
            False
        )

        if not isinstance(save_image, bool):

            return jsonify({

                'status': 'error',

                'message': 'save_image 必须是布尔值'

            }), 400

        # 1. 解码
        img = decode_image(
            image_data
        )

        # 2. 推理
        df = run_inference(
            img
        )

        # 3. 绘制检测框
        pil_img, detections = draw_detections(
            img,
            df
        )

        result_base64 = ''

        # 4. 手动抓拍时保存结果图片
        if save_image:

            result_base64 = image_to_base64(
                pil_img
            )

            save_path = save_pil_image(
                pil_img,
                prefix='result'
            )

            clean_old_images()

            print('抓拍图片已保存:', save_path)

        # 5. 返回结果
        print(
            f'检测完成，目标数量: {len(detections)}'
        )

        return jsonify({

            'status': 'success',

            'detections': detections,

            'result_image': result_base64

        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({

            'status': 'error',

            'message': str(e)

        }), 500


# ==================================================
# 15. 原图同步接口
# ==================================================

@app.route('/sync_raw', methods=['POST'])
def sync_raw():

    try:

        data = request.get_json(
            silent=True
        )

        if not isinstance(data, dict):

            return jsonify({

                'status': 'error',

                'message': '请求数据格式错误'

            }), 400

        img = decode_image(
            data.get('image')
        )

        timestamp = time.time_ns() // 1_000_000

        filename = f'raw_{timestamp}.jpg'

        save_path = os.path.join(
            SHARED_DIR,
            filename
        )

        success = cv2.imwrite(
            save_path,
            img
        )

        if not success:

            raise IOError('原图保存失败')

        print('原图已保存:', save_path)

        return jsonify({

            'status': 'success',

            'message': '原图同步成功',

            'filename': filename

        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({

            'status': 'error',

            'message': str(e)

        }), 500


# ==================================================
# 16. 请求体过大处理
# ==================================================

@app.errorhandler(413)
def request_too_large(e):

    return jsonify({

        'status': 'error',

        'message': '图片过大，请控制在16MB以内'

    }), 413


# ==================================================
# 17. 启动服务
# ==================================================

if __name__ == '__main__':

    print('\n' + '=' * 50)

    print('YOLOv5 道路病害智能检测系统启动成功')

    print('访问地址: http://192.168.20.30:5000')

    print('模型路径:', MODEL_PATH)

    print('推理设备:', device)

    print('输入尺寸:', IMAGE_SIZE)

    print('置信度阈值:', CONF_THRESHOLD)

    print('IOU 阈值:', IOU_THRESHOLD)

    print('=' * 50 + '\n')

    app.run(

        host='0.0.0.0',

        port=5000,

        debug=False,

        threaded=True

    )
