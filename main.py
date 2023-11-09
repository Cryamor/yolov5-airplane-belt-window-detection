from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, LoadScreenshots
from utils.general import *
from utils.plots import Annotator, save_one_box, colors
from utils.torch_utils import select_device, smart_inference_mode
from ui import Ui_mainWindow

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)
    # emit：detecting/pause/stop/finished/error msg
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = ROOT / 'weights/best-sort.pt'
        self.current_weights = ROOT / 'weights/best.pt'
        self.data = ROOT / 'belt/belt_parameter.yaml'
        self.source = 0  # 0 for webcam
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        # self.max_det = 1000  # maximum detections per image
        # self.classes = None  # filter by class: --class 0, or --class 0 2 3
        # self.agnostic_nms = False  # class-agnostic NMS
        self.jump_out = False  # jump out of loop
        self.is_continue = True  # continue/pause
        self.percent_length = 1000  # progress bar
        self.rate_check = True  # enable delay
        self.rate = 100
        self.save_fold = ROOT / 'result'
        # self.project = ROOT / 'runs/detect'
        # self.name = 'exp'  # save results to project/name
        # self.exist_ok = False  # existing project/name ok, do not increment
        # self.save_txt = False  # save results to *.txt


    @smart_inference_mode()
    def run(self,
            imgsz=(640, 640),  # inference size (height, width)
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=1,  # video frame-rate stride
            ):
        try:
            source = str(self.source)
            save_img = not nosave and not source.endswith('.txt')  # save inference images
            is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
            is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
            webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
            screenshot = source.lower().startswith('screen')
            if is_url and is_file:
                source = check_file(source)  # download

            # Directories
            save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
            (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

            # Load model
            device = select_device(device)
            model = DetectMultiBackend(self.weights, device=device, dnn=dnn, data=self.data, fp16=half)
            stride, names, pt = model.stride, model.names, model.pt
            imgsz = check_img_size(imgsz, s=stride)  # check image size

            # Dataloader
            bs = 1  # batch_size
            if webcam:
                view_img = check_imshow(warn=True)
                dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
                bs = len(dataset)
                print(bs)
            elif screenshot:
                dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            vid_path, vid_writer = [None] * bs, [None] * bs

            # Run inference
            model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
            seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

            count = 0
            jump_count = 0
            start_time = time.time()
            dataset = iter(dataset)
            while True:
                if self.is_continue:
                    path, im, im0s, self.vid_cap, s = next(dataset)
                    count += 1
                    if count % 30 == 0 and count >= 30:
                        fps = int(30 / (time.time() - start_time))
                        self.send_fps.emit('fps：' + str(fps))
                        start_time = time.time()
                    if self.vid_cap:
                        percent = int(count / self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) * self.percent_length)
                        self.send_percent.emit(percent)
                    else:
                        percent = self.percent_length
                    statistic_dic = {name: 0 for name in names.values()}

                    with dt[0]:
                        im = torch.from_numpy(im).to(model.device)
                        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                        im /= 255  # 0 - 255 to 0.0 - 1.0
                        if len(im.shape) == 3:
                            im = im[None]  # expand for batch dim
                    # Inference
                    with dt[1]:
                        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                        pred = model(im, augment=augment, visualize=visualize)
                    # NMS
                    with dt[2]:
                        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms, max_det=max_det)
                    # Second-stage classifier (optional)
                    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                    # Process predictions
                    for i, det in enumerate(pred):  # per image
                        seen += 1
                        if webcam:  # batch_size >= 1
                            p, im0, frame = path[i], im0s[i].copy(), dataset.count
                            s += f'{i}: '
                        else:
                            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                        p = Path(p)  # to Path
                        save_path = str(save_dir / p.name)  # im.jpg
                        txt_path = str(save_dir / 'labels' / p.stem) + (
                            '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                        s += '%gx%g ' % im.shape[2:]  # print string
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        imc = im0.copy() if save_crop else im0  # for save_crop
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, 5].unique():
                                n = (det[:, 5] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                if save_txt:  # Write to file
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                        -1).tolist()  # normalized xywh
                                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                    with open(f'{txt_path}.txt', 'a') as f:
                                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                if save_img or save_crop or view_img:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    statistic_dic[names[c]] += 1
                                    label = None if hide_labels else (
                                        names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                    annotator.box_label(xyxy, label, color=colors(c, True))

                                if save_crop:
                                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                                                 BGR=True)

                        # Stream results
                        im0 = annotator.result()
                        # if view_img:
                        #     if platform.system() == 'Linux' and p not in windows:
                        #         windows.append(p)
                        #         cv2.namedWindow(str(p),
                        #                         cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        #         cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                        #     cv2.imshow(str(p), im0)
                        #     cv2.waitKey(1)  # 1 millisecond
                            # Save results (image with detections)
                        if save_img:
                            if dataset.mode == 'image':
                                cv2.imwrite(save_path, im0)

                            else:  # 'video' or 'stream'
                                if vid_path[i] != save_path:  # new video
                                    vid_path[i] = save_path
                                    if isinstance(vid_writer[i], cv2.VideoWriter):
                                        vid_writer[i].release()  # release previous video writer
                                    if self.vid_cap:  # video
                                        fps = self.vid_cap.get(cv2.CAP_PROP_FPS)
                                        w = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                        h = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    else:  # stream
                                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                                    save_path = str(
                                        Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                                                    (w, h))
                                vid_writer[i].write(im0)

                        # Print time (inference-only)
                        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

                    if self.rate_check:
                        time.sleep(1 / self.rate)
                    im0 = annotator.result()

                    self.send_img.emit(im0)
                    self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                    self.send_statistic.emit(statistic_dic)
                    if self.save_fold:
                        os.makedirs(self.save_fold, exist_ok=True)
                        if self.vid_cap is None:
                            save_path = os.path.join(self.save_fold,
                                                     time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                   time.localtime()) + '.jpg')
                            cv2.imwrite(save_path, im0)
                        else:
                            if count == 1:
                                ori_fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
                                if ori_fps == 0:
                                    ori_fps = 25
                                # width = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                # height = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                width, height = im0.shape[1], im0.shape[0]
                                save_path = os.path.join(self.save_fold,
                                                         time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + '.mp4')
                                self.out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), ori_fps,
                                                           (width, height))
                            self.out.write(im0)
                    if percent == self.percent_length and not webcam:
                        print(f"count: {count}")
                        self.send_percent.emit(0)
                        self.send_msg.emit('Finished')
                        if hasattr(self, 'out'):
                            self.out.release()
                        break

                if self.jump_out:  # stop
                    # self.vid_cap.release()
                    self.send_percent.emit(0)
                    self.send_msg.emit('stop')
                    if hasattr(self, 'out'):
                        self.out.release()
                    break
                if self.current_weights != self.weights:  # change model
                    pass

            # Print results
            t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
            if save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
            if update:
                strip_optimizer(self.weights[0])  # update model (to fix SourceChangeWarning)

        except Exception as e:
            self.send_msg.emit(f'{e}')
            print(e)

class MainWindow(QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False
        
        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        # self.qtimer.timeout.connect(lambda: self.statistic_label.clear())
        
        # yolov5 thread
        self.det_thread = DetThread()
        self.det_thread.percent_length = self.progressBar.maximum()
        self.det_thread.send_raw.connect(lambda x: self.show_img(x, self.raw_video))
        self.det_thread.send_img.connect(lambda x: self.show_img(x, self.out_video))
        self.det_thread.send_statistic.connect(self.show_statistic)
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))
        
        self.fileButton.clicked.connect(self.open_file)
        self.cameraButton.clicked.connect(self.open_cam)
        # self.rtspButton.clicked.connect(self.chose_rtsp)

        self.runButton.clicked.connect(self.run_or_continue)
        self.stopButton.clicked.connect(self.stop)

        # self.comboBox.currentTextChanged.connect(self.change_model)
        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))
        # self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        # self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))

        # self.checkBox.clicked.connect(self.checkrate)
        self.saveCheckBox.clicked.connect(self.is_save)
        self.saveCheckBox.setChecked(True)
        self.load_settings()
        
    def open_cam(self):
        source = 0
        # source = 1
        self.det_thread.source = source
        self.stop()
        self.status_msg(f'Loading camera: {self.det_thread.source}')

    def load_settings(self):
        '''load from .json'''
        # not done
        iou = 0.45
        conf = 0.25
        rate = 10
        check = 0
        savecheck = 0
        self.confSpinBox.setValue(conf)
        self.iouSpinBox.setValue(iou)
        # self.rateSpinBox.setValue(rate)
        # self.checkBox.setCheckState(check)
        self.det_thread.rate_check = check
        self.saveCheckBox.setCheckState(savecheck)
        self.is_save()
        
    def is_save(self):
        if self.saveCheckBox.isChecked():
            self.det_thread.save_fold = ROOT / 'result'
        else:
            self.det_thread.save_fold = None

    def open_file(self):
        default_folder = './belt/datasets/images'
        filename, _ = QFileDialog.getOpenFileName(self,
                                                  'Video/Image',
                                                  default_folder,
                                                  "*.mp4 *.mkv *.avi *.flv *.jpg *.png *.jpeg"
                                                  )
        if filename:
            self.det_thread.source = filename
            self.status_msg(f'Loading file：{filename} ...')
            self.stop()
            try:
                if filename.endswith(('png', 'jpg', 'jpeg')):
                    im = cv2.imread(filename)
                    self.show_img(im, self.raw_video)
                    self.out_video.clear()
                else:
                    vid = cv2.VideoCapture(filename)
                    _, frame = vid.read()
                    self.show_img(frame, self.raw_video)
                    self.out_video.clear()
            except Exception as e:
                print(e)
            self.status_msg(f'Loading file：{filename} ... Done.')

    def show_img(self, img, label):
        try:
            i_h, i_w, _ = img.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if i_w/w > i_h/h:
                scal = w / i_w
                n_w = w
                n_h = int(scal * i_h)
                img1 = cv2.resize(img, (n_w, n_h))
            else:
                scal = h / i_h
                n_h = h
                n_w = int(scal * i_w)
                img1 = cv2.resize(img, (n_w, n_h))

            frame = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))
    
    def show_statistic(self, statistic_dic):
        try:
            self.resultWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' '+str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            self.resultWidget.addItems(results)
        except Exception as e:
            print(repr(e))
    
    def status_msg(self, msg):
        self.statusMsgBar.showMessage(msg)
    
    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.status_msg(msg)
        if msg == 'Finished':
            self.saveCheckBox.setEnabled(True)

    def checkrate(self):
        if self.checkBox.isChecked():
            self.det_thread.rate_check = True
        else:
            self.det_thread.rate_check = False
    
    def change_val(self, x, flag):
        '''change value of conf_thre, iou_thre, ...'''
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x*100))
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x/100)
            self.det_thread.conf_thres = x/100
        elif flag == 'iouSpinBox':
            self.iouSlider.setValue(int(x*100))
        elif flag == 'iouSlider':
            self.iouSpinBox.setValue(x/100)
            self.det_thread.iou_thres = x/100
        # elif flag == 'rateSpinBox':
        #     self.rateSlider.setValue(x)
        # elif flag == 'rateSlider':
        #     self.rateSpinBox.setValue(x)
        #     self.det_thread.rate = x * 10
        else:
            pass   
  
    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()
            
    def stop(self):
        self.det_thread.jump_out = True
        self.saveCheckBox.setEnabled(True) 
    
    def run_or_continue(self):
        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.saveCheckBox.setEnabled(False)
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
            source = str(self.det_thread.source)
            if source.isnumeric():
                source = f'camera: {source}'
            self.status_msg(f'Detecting... model：{os.path.basename(self.det_thread.weights)}，source：{source}')
        else:
            self.det_thread.is_continue = False
            self.status_msg('Pause')
           
    # def close(self):
    #     self.det_thread.jump_out = True
    #     #
    #     sys.exit(0)
    #
    # def mousePressEvent(self, event):
    #     self.m_Position = event.pos()
    #     if event.button() == Qt.LeftButton:
    #         if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
    #                 0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
    #             self.m_flag = True
    #
    # def mouseMoveEvent(self, QMouseEvent):
    #     if Qt.LeftButton and self.m_flag:
    #         self.move(QMouseEvent.globalPos() - self.m_Position)
    #
    # def mouseReleaseEvent(self, QMouseEvent):
    #     self.m_flag = False
    #
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MainWindow()
    myWindow.show()
    sys.exit(app.exec_())