#!/usr/bin/env python3

import os
import sys

from pathlib import Path
from numpy import source

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()

ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from YOLOv5_ROS.msg import BoundingBox, BoundingBoxes
from rostopic import get_topic_type


@torch.no_grad()
class Detector:
    def __init__(self):
        # # param_name = rospy.search_param('global_example')
        # # v = rospy.get_param_names()
        # # print(v)
        # self.weights=rospy.get_param("~weights", "weights/yolov5s.pt")

        self.weights=ROOT / 'weights/yolov5s.pt'  # model.pt path(s)
        self.source=ROOT / 'data/images'  # file/dir/URL/glob, 0 for webcam
        # self.source=0
        self.data=ROOT / 'data/coco128.yaml'  # dataset.yaml path
        self.imgsz=(640, 640)  # inference size (height, width)
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img=False  # show results
        self.save_txt=False  # save results to *.txt
        self.save_conf=False  # save confidences in --save-txt labels
        self.save_crop=False  # save cropped prediction boxes
        self.nosave=True  # do not save images/videos
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.visualize=False  # visualize features
        self.update=False  # update all models
        self.project=ROOT / 'runs/detect'  # save results to project/name
        self.name='exp'  # save results to project/name
        self.exist_ok=False  # existing project/name ok, do not increment
        self.line_thickness=3  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.half=False  # use FP16 half-precision inference
        self.dnn=False  # use OpenCV DNN for ONNX inference

        # self.weights = rospy.get_param("~weights", "/home/hoorad/catkin_ws/src/yolo5/weights/yolov5s.pt")                                # model.pt path(s)
        # self.source = rospy.get_param("~source", "/home/hoorad/catkin_ws/src/yolo5/data/images") 
        # print(self.source)                                 # file/dir/URL/glob, 0 for webcam
        # self.data = rospy.get_param("~data", "data/coco128.yaml")                                      # dataset.yaml path
        # self.imgsz = (rospy.get_param("~imgsz_h", "640"), rospy.get_param("~imgsz_w", "640"))   # inference size (height, width)
        # self.conf_thres = rospy.get_param("~conf_thres")                          # confidence threshold
        # self.iou_thres = rospy.get_param("~iou_thres", "0.45")                            # NMS IOU threshold
        # self.max_det = rospy.get_param("~max_det", "1000")                                # maximum detections per image
        # self.device = rospy.get_param("~device", "")                                  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        # self.view_img = rospy.get_param("~view_img", "False")                              # show results
        # self.save_txt = rospy.get_param("~save_txt", "False")                              # save results to *.txt
        # self.save_conf = rospy.get_param("~save_conf", "False")                            # save confidences in --save-txt labels
        # self.save_crop = rospy.get_param("~save_crop", "False")                            # save cropped prediction boxes
        # self.nosave = rospy.get_param("~nosave", "True")                                  # do not save images/videos
        # self.classes = rospy.get_param("~classes", "None")                                # filter by class: --class 0, or --class 0 2 3
        # self.agnostic_nms = rospy.get_param("~agnostic_nms", "False")                      # class-agnostic NMS
        # self.augment = rospy.get_param("~augment", "False")                                # augmented inference
        # self.visualize = rospy.get_param("~visualize", "False")                            # visualize features
        # self.update = rospy.get_param("~update", "False")                                  # update all models
        # self.project = rospy.get_param("~project", "runs/detect")                                # save results to project/name
        # self.name = rospy.get_param("~name", "exp")                                      # save results to project/name
        # self.exist_ok = rospy.get_param("~exist_ok", "False")                              # existing project/name ok, do not increment
        # self.line_thickness = rospy.get_param("~line_thickness", "3")                  # bounding box thickness (pixels)
        # self.hide_labels = rospy.get_param("~hide_labels", "False")                        # hide labels
        # self.hide_conf = rospy.get_param("~hide_conf", "False")                            # hide confidences
        # self.half = rospy.get_param("~half", "False")                                      # use FP16 half-precision inference
        # self.dnn = rospy.get_param("~dnn", "False")                                        # use OpenCV DNN for ONNX inference

        self.source = str(self.source)
        self.save_img = not self.nosave and not self.source.endswith('.txt')  # save inference images
        self.is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        self.is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or (self.is_url and not self.is_file)
        if self.is_url and self.is_file:
            self.source = check_file(self.source)  # download
        
        # Directories
        self.save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

        # Dataloader
        if self.webcam:
            self.view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
            self.bs = len(self.dataset)  # batch_size
        else:
            self.dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
            self.bs = 1  # batch_size
        self.vid_path, self.vid_writer = [None] * self.bs, [None] * self.bs

        # Half
        self.half = rospy.get_param("~half", False)
        self.half &= (
            self.pt or self.jit or self.onnx or self.engine
        ) and self.device.type != "cpu"  # FP16 supported on limited backends with CUDA
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()
        bs = 1  # batch_size
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model.warmup(imgsz=(1 if self.pt else self.bs, 3, self.imgsz[0], self.imgsz[1]))  # warmup        
        

        # Initialize subscriber to Image/CompressedImage topic
        input_image_type, input_image_topic, _ = get_topic_type("/cv_camera/image_raw", blocking = True)
        self.compressed_input = input_image_type == "sensor_msgs/CompressedImage"

        if self.compressed_input:
            self.image_sub = rospy.Subscriber(input_image_topic, CompressedImage, self.callback, queue_size=1)

        else:
            self.image_sub = rospy.Subscriber(input_image_topic, Image, self.callback, queue_size=1)

        # Initialize prediction publisher
        self.pred_pub = rospy.Publisher("/yolov5/detections", BoundingBoxes, queue_size=10)

        # Initialize image publisher
        self.publish_image = rospy.get_param("~publish_image", "False")
        if self.publish_image:
            self.image_pub = rospy.Publisher("/yolov5/image_out", Image, queue_size=10)
        
        # Initialize CV_Bridge
        self.bridge = CvBridge()
        print("caaaaaaaaaaaaaaall")


    def run(self):
        # Run inference
        # Edited
        # model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        # self.model.warmup(imgsz=(1 if self.pt else self.bs, 3, self.imgsz[0], self.imgsz[1]))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in self.dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            self.visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.visualize else False
            pred = self.model(im, augment=self.augment, visualize=self.visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if self.webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), self.dataset.count
                    # Edited
                    # s += f'{i}: '
                    s += ('{i}: ').format(i = i)
                else:
                    p, im0, frame = path, im0s.copy(), getattr(self.dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # im.jpg
                # Edited
                # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else ('_{frame}').format(frame = frame))  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        # Edited
                        # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        s += ("{n} {names}{s}, ").format(n = n, names = self.names[int(c)], s = 's' * (n > 1))  # add to string

                    # Edited
                    for i, tmp in enumerate(reversed(det)):
                        # print(len(reversed(det)))
                        xyxy = []
                        for i, det_element in enumerate(tmp):
                            if i < 4:
                                xyxy.append(det_element)
                            elif i == 4:
                                conf = det_element
                            elif i == 5:
                                cls = det_element

                        # To-Do
                        # if save_txt:  # Write to file
                        #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        #     # Edited
                        #     # line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        #     with open(txt_path + '.txt', 'a') as f:
                        #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            # Edited
                            # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            label = None if self.hide_labels else (self.names[c] if self.hide_conf else ('{names} {conf:.2f}').format(names = self.names[c], conf = conf))
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            if self.save_crop:
                                # Edited
                                # save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                                save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c] / ('{stem}.jpg').format(stem = p.stem), BGR=True)

                    # for *xyxy, conf, cls in reversed(det):
                    #     if save_txt:  # Write to file
                    #         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #         line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    #         with open(txt_path + '.txt', 'a') as f:
                    #             f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    #     if save_img or save_crop or view_img:  # Add bbox to image
                    #         c = int(cls)  # integer class
                    #         label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    #         annotator.box_label(xyxy, label, color=colors(c, True))
                    #         if save_crop:
                    #             save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    
                    

                # Stream results
                im0 = annotator.result()
                if self.view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if self.save_img:
                    if self.dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if self.vid_path[i] != save_path:  # new video
                            self.vid_path[i] = save_path
                            if isinstance(self.vid_writer[i], cv2.VideoWriter):
                                self.vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            self.vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        self.vid_writer[i].write(im0)

            # Print time (inference-only)
            # Edited
            # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
            LOGGER.info(('{s}Done. ({dt:.3f}s)').format(s = s, dt = t3 - t2))

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        # Edited
        # To-Do: edit logger
        # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if self.save_txt or self.save_img:
            # Edited
            # s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            # LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
            s = ("\n{len} labels saved to {dir}").format(len = len(list(self.save_dir.glob('labels/*.txt'))), dir = self.save_dir / 'labels') if self.save_txt else ''
            LOGGER.info(("Results saved to {cstr}{s}").format(cstr = colorstr('bold', self.save_dir), s = s))
        if self.update:
            strip_optimizer(self.weights)  # update model (to fix SourceChangeWarning)



    def callback(self, data):
        """adapted from yolov5/detect.py"""
        print("caaaaaaaaaaaaaaall")
        if self.compressed_input:
            im = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        else:
            im = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        
        im, im0 = self.preprocess(im)
        # print(im.shape)
        # print(img0.shape)
        # print(img.shape)

        # Run inference
        im = torch.from_numpy(im).to(self.device) 
        im = im.half() if self.half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det
        )

        ### To-do move pred to CPU and fill BoundingBox messages
        
        # Process predictions 
        det = pred[0].cpu().numpy()

        bounding_boxes = BoundingBoxes()
        bounding_boxes.header = data.header
        bounding_boxes.image_header = data.header
        
        annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                bounding_box = BoundingBox()
                c = int(cls)
                # Fill in bounding box message
                bounding_box.Class = self.names[c]
                bounding_box.probability = conf 
                bounding_box.xmin = int(xyxy[0])
                bounding_box.ymin = int(xyxy[1])
                bounding_box.xmax = int(xyxy[2])
                bounding_box.ymax = int(xyxy[3])

                bounding_boxes.bounding_boxes.append(bounding_box)

                # Annotate the image
                if self.publish_image or self.view_image:  # Add bbox to image
                      # integer class
                    label = f"{self.names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))       

                
                ### POPULATE THE DETECTION MESSAGE HERE

            # Stream results
            im0 = annotator.result()

        # Publish prediction
        self.pred_pub.publish(bounding_boxes)

        # Publish & visualize images
        if self.view_image:
            cv2.imshow(str(0), im0)
            cv2.waitKey(1)  # 1 millisecond
        if self.publish_image:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(im0, "bgr8"))
        

    def preprocess(self, img):
        """
        Adapted from yolov5/utils/datasets.py LoadStreams class
        """
        img0 = img.copy()
        img = np.array([letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]])
        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return img, img0




def main():
    obj = Detector()
    # obj.run()


if __name__ == "__main__":
    main()