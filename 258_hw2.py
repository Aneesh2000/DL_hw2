import os
import time
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from ModelDev.multimodels import MultiModels

VIDEO_PATH = "Users/aneesh/downloads/object (16).mp4"
SKIP_FRAMES = 5
NUM_BENCHMARK = 100
INPUT_SIZE = (640, 640)

ONNX_YOLO_PATH = "yolov5s.onnx"
ONNX_FRCNN_PATH = "fasterrcnn.onnx"
TRT_YOLO_ENGINE = "yolov5s.trt"
TRT_FRCNN_ENGINE = "fasterrcnn.trt"

TRAIN_ANNOT_JSON = "Users/aneesh/downloads/annotations.json"
TRAIN_IMG_DIR = "Users/aneesh/downloads/images/"
BATCH_SIZE = 4
NUM_EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CocoLikeDataset(Dataset):
    def __init__(self, img_dir, ann_json, transform=None):
        self.records = []
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = cv2.imread(os.path.join(self.img_dir, rec['img']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        target = {
            'boxes': rec['boxes'].to(DEVICE),
            'labels': rec['labels'].to(DEVICE)
        }
        return img, target

def frame_generator(video_path, skip=1):
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % skip == 0:
            yield frame
        idx += 1
    cap.release()

def train():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(INPUT_SIZE),
    ])
    ds = CocoLikeDataset(TRAIN_IMG_DIR, TRAIN_ANNOT_JSON, transform=tf)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    model = MultiModels(backbone='resnet50', num_classes=80).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for imgs, targets in loader:
            imgs = torch.stack(imgs).to(DEVICE)
            outputs = model(imgs)
            labels = torch.cat([t['labels'] for t in targets])
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg = running_loss / len(loader)
        print(f"[Epoch {epoch}/{NUM_EPOCHS}] loss: {avg:.4f}")

    torch.save(model.state_dict(), "trained_multimodels.pth")
    print("Training complete. Model saved to trained_multimodels.pth")

class ONNXDetector:
    def __init__(self, model_path):
        self.sess = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def preprocess(self, img):
        im = cv2.resize(img, INPUT_SIZE)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) / 255.0
        im = np.transpose(im, (2,0,1))[None].astype(np.float32)
        return im

    def infer(self, img):
        im = self.preprocess(img)
        start = time.time()
        outputs = self.sess.run(None, {self.input_name: im})
        return outputs, time.time() - start

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for b in range(engine.num_bindings):
        name = engine.get_binding_name(b)
        shape = engine.get_binding_shape(b)
        size = trt.volume(shape) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(name))
        host_mem = cuda.pagelocked_empty(size, dtype)
        dev_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(dev_mem))
        if engine.binding_is_input(name):
            inputs.append((host_mem, dev_mem))
        else:
            outputs.append((host_mem, dev_mem))
    return inputs, outputs, bindings, stream

class TRTDetector:
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)

    def preprocess(self, img):
        im = cv2.resize(img, INPUT_SIZE)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) / 255.0
        return np.transpose(im, (2,0,1)).ravel().astype(np.float32)

    def infer(self, img):
        in_h, in_d = self.inputs[0]
        in_h[:] = self.preprocess(img)
        cuda.memcpy_htod_async(in_d, in_h, self.stream)
        start = time.time()
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        outs = []
        for out_h, out_d in self.outputs:
            cuda.memcpy_dtoh_async(out_h, out_d, self.stream)
            outs.append(out_h.copy())
        self.stream.synchronize()
        return outs, time.time() - start

def benchmark_all():
    onnx_yolo = ONNXDetector(ONNX_YOLO_PATH)
    onnx_frcnn = ONNXDetector(ONNX_FRCNN_PATH)
    trt_yolo = TRTDetector(TRT_YOLO_ENGINE)
    trt_frcnn = TRTDetector(TRT_FRCNN_ENGINE)

    setups = [
        ("ONNX YOLO", onnx_yolo),
        ("TensorRT YOLO", trt_yolo),
        ("ONNX FRCNN", onnx_frcnn),
        ("TensorRT FRCNN", trt_frcnn),
    ]

    print(f"Benchmarking on video: {VIDEO_PATH}")
    for name, det in setups:
        gen = frame_generator(VIDEO_PATH, skip=SKIP_FRAMES)
        latencies = []
        for i, frame in enumerate(gen):
            _, t = det.infer(frame)
            latencies.append(t)
            if i+1 >= NUM_BENCHMARK:
                break
        m, s = np.mean(latencies), np.std(latencies)
        print(f"{name}: {m*1000:.1f} ± {s*1000:.1f} ms/frame")

if __name__ == "__main__":
    train()
    benchmark_all()
