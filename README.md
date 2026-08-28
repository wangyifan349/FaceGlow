# FaceGlow

FaceGlow 是一个用于**离线照片美颜与批量处理**的轻量 Python 项目。它不调用摄像头，也不处理实时视频。InsightFace 负责检测人脸并提供关键点，OpenCV 与 NumPy 负责实际图像处理，包括精确肤色区域识别、五官保护、保边磨皮、纹理恢复、局部美白以及脸颊和下颌的局部瘦脸形变。

程序支持单张图片、整个文件夹以及递归子目录批量处理，也可以作为 Python 模块导入其他项目。输入照片始终只读，输出结果写入新的文件或目录，程序不会覆盖、移动或删除原始照片。

## 主要功能

- InsightFace `buffalo_l` 人脸检测。
- InsightFace 106 点人脸关键点定位。
- InsightFace 5 点关键点辅助五官保护。
- YCrCb + HSV + Lab 多颜色空间肤色判断。
- 106 点几何脸区与肤色区域联合约束。
- 眉毛、眼睛、嘴唇、鼻部中央细节保护。
- 双边滤波保边磨皮。
- 高频皮肤纹理回填，减少塑料感。
- Lab 局部亮度提升与轻微色偏中和。
- Gamma 局部美白。
- 五官局部细节恢复。
- 左右脸颊与左右下颌独立局部瘦脸。
- 单张照片处理。
- 文件夹批量处理。
- 递归处理子文件夹。
- 保持原目录结构输出。
- 多人脸处理，默认处理检测到的全部人脸。
- 输出文件自动避让同名文件，不覆盖已有结果。
- 支持 Windows Unicode 路径。
- 核心算法均拆分为独立 `def`，方便单独调用、测试和替换。

## 项目结构

```text
FaceGlow/
├── beauty_processor.py
├── README.md
└── LICENSE
```

## 环境建议

建议使用 Python 3.10、3.11 或 3.12。

主要依赖：

```text
insightface
onnxruntime
opencv-python
numpy
```

默认使用 `CPUExecutionProvider`，因此不要求 NVIDIA CUDA 环境。

## 安装

克隆项目：

```bash
git clone https://github.com/wangyifan349/FaceGlow
cd FaceGlow
```

安装依赖：

```bash
pip install insightface onnxruntime opencv-python numpy
```

第一次运行时，InsightFace 可能需要下载 `buffalo_l` 模型文件。

## 快速开始

### 处理单张照片

```bash
python beauty_processor.py --input photo.jpg --output beauty_output
```

例如输入：

```text
photo.jpg
```

会生成类似：

```text
beauty_output/photo_beauty.jpg
```

原始 `photo.jpg` 不会被修改。

### 指定单张输出文件

```bash
python beauty_processor.py --input photo.jpg --output result.jpg
```

如果 `result.jpg` 已存在，程序不会覆盖它，而会自动选择新的名称，例如：

```text
result_1.jpg
result_2.jpg
```

### 批量处理文件夹

```bash
python beauty_processor.py --input photos --output beauty_output
```

默认递归扫描子目录。

例如输入目录：

```text
photos/
├── portrait.jpg
├── group.png
└── trip/
    └── image.jpg
```

输出目录会保持相对目录结构：

```text
beauty_output/
├── portrait_beauty.jpg
├── group_beauty.png
└── trip/
    └── image_beauty.jpg
```

### 不递归处理子目录

```bash
python beauty_processor.py --input photos --output beauty_output --no-recursive
```

## 调整美颜强度

```bash
python beauty_processor.py \
  --input photos \
  --output beauty_output \
  --smooth 0.62 \
  --whiten 0.24 \
  --slim 0.16 \
  --detail 0.30
```

主要参数：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--smooth` | `0.62` | 磨皮强度，建议 `0.0 ~ 1.0` |
| `--whiten` | `0.24` | 局部美白强度，建议 `0.0 ~ 1.0` |
| `--slim` | `0.16` | 瘦脸强度，建议 `0.0 ~ 1.0` |
| `--detail` | `0.30` | 皮肤纹理与五官细节恢复强度 |
| `--max-faces` | `0` | 最大处理人脸数，`0` 表示全部 |
| `--det-size` | `640` | InsightFace 检测输入尺寸 |
| `--det-threshold` | `0.50` | 人脸检测阈值 |
| `--output-suffix` | `_beauty` | 输出文件名追加的后缀 |
| `--no-recursive` | 关闭 | 不扫描子目录 |

例如只处理面积最大的 1 张脸：

```bash
python beauty_processor.py \
  --input photo.jpg \
  --output beauty_output \
  --max-faces 1
```

## 图像处理流程

程序不会直接对整张图片粗暴套滤镜，而是先确定人脸和真正需要处理的皮肤区域。

### 1. 人脸检测

InsightFace 检测图片中的人脸，并获取每张脸的边界框。

### 2. 106 点关键点定位

使用 `landmark_2d_106` 获取人脸关键点，为脸部几何区域和瘦脸位置提供依据。

### 3. 几何脸区

`create_geometry_face_mask()` 使用 106 点凸包与保守的人脸椭圆相交，减少头发、背景以及脸外区域进入美颜范围。

### 4. 多颜色空间肤色检测

`create_skin_color_mask()` 同时使用：

- YCrCb
- HSV
- Lab

三个颜色空间进行肤色判断，并通过投票方式融合，不依赖单一阈值。

### 5. 五官保护

`create_feature_protection_mask()` 根据 5 点关键点保护：

- 左右眼睛
- 左右眉毛附近
- 嘴唇
- 鼻部中央细节

这样磨皮不会直接把眉眼和嘴唇边缘磨糊。

### 6. 精确皮肤 Mask

`create_precise_skin_mask()` 将几何脸区、肤色区域和五官保护区域组合，再进行边缘羽化。

最终磨皮和美白主要作用于真正的脸部皮肤区域，而不是整张照片。

### 7. 磨皮

`smooth_skin()` 使用双边滤波进行保边平滑，然后从原图提取高频纹理并按比例重新回填。

这样可以降低皮肤噪声和细小瑕疵，同时尽量保留原有皮肤纹理。

### 8. 美白

`whiten_skin()` 主要包含：

- Lab 明度提升。
- 根据剩余高光空间控制提亮量。
- Lab 色度轻微向中性方向靠近。
- 柔和 Gamma 修正。
- 使用精确皮肤 Mask 与原图混合。

因此背景、头发和大部分五官不会被一起强制提亮。

### 9. 五官细节恢复

`restore_feature_details()` 对五官和轮廓保护区域进行轻量反锐化，恢复磨皮之后的眉眼、嘴唇和轮廓细节。

### 10. 瘦脸

`slim_face()` 不直接缩放整张脸，而是分别寻找：

- 左脸颊
- 右脸颊
- 左下颌
- 右下颌

然后使用 `local_translation_warp()` 做带径向衰减的局部逆向映射，使脸部边缘轻微向中心收缩。

几何形变放在磨皮、美白和细节恢复之后执行，避免提前形变导致关键点和 Mask 错位。

## 作为 Python 模块使用

建议创建一次 `BeautyProcessor` 后重复使用。这样批量处理时 InsightFace 模型只需要初始化一次。

```python
from beauty_processor import BeautyConfig, BeautyProcessor

config = BeautyConfig(
    smooth_strength=0.62,
    whiten_strength=0.24,
    slim_strength=0.16,
    detail_strength=0.30,
    max_faces=0,
)

processor = BeautyProcessor(config)
```

### 处理 NumPy / OpenCV 图像

```python
import cv2

from beauty_processor import BeautyProcessor

processor = BeautyProcessor()
source_image = cv2.imread("photo.jpg")
beautified_image, processed_face_count = processor.beautify_image(source_image)

print(processed_face_count)
```

`beautify_image()` 不会原地修改传入的 NumPy 数组，而是返回新的处理结果。

### 处理单张文件

```python
from beauty_processor import BeautyProcessor, process_image_file

processor = BeautyProcessor()
processing_result = process_image_file(
    input_path="photo.jpg",
    output="beauty_output",
    processor=processor,
)

print(processing_result.output_path)
print(processing_result.face_count)
```

### 批量处理文件夹

```python
from beauty_processor import BeautyProcessor, process_batch

processor = BeautyProcessor()
batch_summary = process_batch(
    input_directory="photos",
    output_directory="beauty_output",
    processor=processor,
    recursive=True,
)

print(batch_summary.total_files)
print(batch_summary.success_files)
```

### 文件或目录统一入口

```python
from beauty_processor import BeautyProcessor, process_path

processor = BeautyProcessor()
processing_result = process_path(
    input_path="photos",
    output="beauty_output",
    processor=processor,
)
```

## 可独立调用的关键函数

```text
create_geometry_face_mask()       构建 106 点几何脸区
create_skin_color_mask()          多颜色空间肤色识别
create_feature_protection_mask()  构建五官保护区域
create_precise_skin_mask()        构建最终精确皮肤 Mask
blend_with_mask()                 软 Mask 混合
smooth_skin()                     磨皮与高频纹理回填
whiten_skin()                     局部美白
restore_feature_details()         五官细节恢复
choose_side_anchor()              选择脸颊/下颌形变锚点
local_translation_warp()          局部径向形变
slim_face()                       脸颊和下颌瘦脸
beautify_image()                  NumPy 图像函数式入口
process_image_file()              单文件入口
process_batch()                   文件夹批处理入口
process_path()                    文件/目录统一入口
```

这些函数相互分层，后续如果想替换磨皮、美白、肤色检测或瘦脸算法，不需要重新修改批量扫描和文件读写代码。

## 原文件安全

项目默认遵循“不破坏原文件”的原则。

- 输入图片只读取。
- 不删除输入文件。
- 不移动输入文件。
- 不把处理结果直接写回输入文件。
- 输出文件重名时自动选择新的文件名。
- 批量输出目录不能与输入目录完全相同。
- 如果输出目录位于输入目录内部，本次扫描会自动排除该输出目录，避免生成文件再次进入同一批任务。
- 批量任务会先完成文件列表扫描，再开始写出结果。

例如下面的命令是推荐方式：

```bash
python beauty_processor.py --input photos --output beauty_output
```

不建议把素材目录本身作为输出位置，程序也会拒绝输入目录和批量输出目录完全相同的情况。

## 输出格式

当前支持：

```text
.jpg
.jpeg
.png
.bmp
.webp
.tif
.tiff
```

默认 JPEG 质量为 `95`，PNG 压缩等级为 `3`。

输出图片会重新编码。项目目前不承诺把原照片中的全部 EXIF、GPS、ICC Profile 或相机厂商私有元数据复制到新文件，但这不会影响原始照片，因为原文件不会被修改。

## 模型与许可证说明

本仓库中**由本项目编写的源代码**使用 MIT License，详见 [LICENSE](LICENSE)。

需要特别注意：MIT License 只覆盖本仓库自身代码，并不会自动改变 InsightFace、ONNX Runtime、OpenCV、NumPy 或第三方预训练模型的许可证。尤其是 InsightFace 预训练模型应按照其官方模型许可和使用条件单独使用。

本项目当前主要面向非商业学习、研究和开源交流场景。如果需要用于商业产品，请先确认所使用 InsightFace 模型及其他第三方组件是否具有对应的商业授权。

## License

MIT License。

Copyright (c) 2026 wangyifan349
