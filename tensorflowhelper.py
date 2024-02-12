from yolov5.detect import run, parse_opt, main

run(
    weights="MELU-Trained-ObjDetection-Model-Yolov5-BEST.pt",
    source="herb_images/im_64.jpg",
    conf_thres=0.4,
    imgsz=(416, 416)
)

