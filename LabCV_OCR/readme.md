### 安装

在包目录下执行

```python
pip3 install -r requirements.txt
```

### 引入

```python
from LabCV_OCR import ocr
```

### 使用

#### 传入图片地址进行 OCR

```python
ocr.ocr_path(path)
```


#### 传入 Pillow 图片进行 OCR

```python
ocr.ocr_pillow(path)
```

#### 传入 Skimage 图片进行 OCR

```python
ocr.ocr_skimage(path)
```

#### 传入 OpenCV 图片进行 OCR

```python
ocr.ocr_opencv(path)
```

### 返回值

返回值是一个二维矩阵，大概是这样的:

[
[识别内容1, 位置1]
[识别内容2, 位置2]
[识别内容3, 位置3]
...
]