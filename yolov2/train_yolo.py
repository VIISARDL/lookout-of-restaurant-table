
import torch

import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import config_yolo as config
import utils_yolo as dt
from datetime import datetime

from pytorchYolo2 import dataset
from pytorchYolo2 import utils as ut
from pytorchYolo2.darknet import Darknet

import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable


class Yolo():

    num_workers = 3
    kwargs = {'num_workers': num_workers, 'pin_memory': True}
    params = []

    def __init__(self, datacfg='', cfgfile='', weightfile=None, namesfile='../Config/atHome.names'):
        self.datacfg    = datacfg
        self.cfgfile    = cfgfile
        self.weightfile = weightfile

        self.__initConfigProvider()

        if self.use_cuda:
            self.model = Darknet(cfgfile).cuda()
        else:
            self.model = Darknet(cfgfile)

        self.region_loss = self.model.loss

        self.model.load_weights(weightfile)

        self.region_loss.seen   = self.model.seen
        self.processed_batches  = self.model.seen / self.cp.batch_size
        self.init_width         = self.model.width
        self.init_height        = self.model.height
        self.init_epoch         = self.model.seen / self.nsamples
        self.namesfile          = namesfile
        self.class_names        =  ut.load_class_names(namesfile)

        self.__init_test_loader()

        self.params_dict = dict(self.model.named_parameters())
        for key, value in self.params_dict.items():
            if key.find('.bn') >= 0 or key.find('.bias') >= 0:
                self.params += [{'params': [value], 'weight_decay': 0.0}]
            else:
                self.params += [{'params': [value],
                                 'weight_decay': self.decay * self.batch_size}]

        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.lr / self.batch_size,
                                   momentum=self.momentum, dampening=0,
                                   weight_decay=self.decay * self.batch_size)

    def adjustLR(self):

        for i in range(len(self.steps)):
            scale = self.scales[i] if i < len(self.scales) else 1
            if self.processed_batches >= self.steps[i]:
                self.lr = self.lr * scale
                if self.processed_batches == self.steps[i]:
                    break
            else:
                break

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr / self.batch_size
        print ("New Learning rate: {}".format(self.lr))

    def train(self, epoch=0, fixLR=0):

        cur_model = self.model

        train_loader = torch.utils.data.DataLoader(
            dataset.listDataset(self.trainDir, shape=(
                self.init_width, self.init_height), shuffle=True,
                transform=transforms.Compose([transforms.ToTensor(), ]),
                train=True, seen=cur_model.seen,
                batch_size=self.batch_size,
                num_workers=self.num_workers),
            batch_size=self.batch_size, shuffle=False, **self.kwargs)

        if fixLR == 0:
            self.adjustLR()
        else:
            self.lr = fixLR

        print('epoch %d, processed %d samples, lr %f' %
              (epoch, epoch * len(train_loader.dataset), self.lr))
        self.model.train()

        for batch_idx, (data, target) in enumerate(train_loader):

            if fixLR == 0:
                self.adjustLR()
            else:
                self.lr = fixLR

            self.processed_batches = self.processed_batches + 1
            if len(torch.nonzero(target)) == 0:
                print ("*" * 60)
                print ("Unknow Error: target return a zeros vector")
                print ("*" * 60)
                continue
            data, target = Variable(data.cuda()), Variable(target)

            self.optimizer.zero_grad()
            output = self.model(data)
            self.region_loss.seen = self.region_loss.seen + data.data.size(0)
            loss = self.region_loss(output, target)
            loss.backward()
            self.optimizer.step()

        if (epoch + 1) % self.save_interval == 0:
            print('save weights to %s/%06d.weights' %
                  (self.backupDir, epoch + 1))
            cur_model.seen = (epoch + 1) * len(train_loader.dataset)
            cur_model.save_weights('%s/%06d.weights' %
                                   (self.backupDir, epoch + 1))

    def test(self, epoch):
        def truths_length(truths):
            for i in range(50):
                if truths[i][1] == 0:
                    return i

        self.model.eval()
        cur_model = self.model
        num_classes = cur_model.num_classes
        anchors = cur_model.anchors
        num_anchors = cur_model.num_anchors
        total = 0.0
        proposals = 0.0
        correct = 0.0

        for batch_idx, (data, target) in enumerate(self.test_loader):
            with torch.no_grad():
                data = Variable(data.cuda())
            output      = self.model(data).data
            all_boxes   = ut.get_region_boxes(
                output, self.conf_thresh, num_classes, anchors, num_anchors)

            for i in range(output.size(0)):
                boxes   = all_boxes[i]
                boxes   = ut.nms(boxes, self.nms_thresh)
                truths  = target[i].view(-1, 5)
                num_gts = truths_length(truths)
                total   = total + num_gts

                for i in range(len(boxes)):
                    if boxes[i][4] > self.conf_thresh:
                        proposals = proposals + 1

                for i in range(num_gts):
                    box_gt = [truths[i][1], truths[i][2], truths[i]
                              [3], truths[i][4], 1.0, 1.0, truths[i][0]]
                    best_iou = 0
                    for j in range(len(boxes)):
                        iou = ut.bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                        best_iou = max(iou, best_iou)

                    if best_iou > self.iou_thresh and boxes[j][6] == box_gt[6].type(torch.LongTensor):
                        correct = correct + 1

        precision = 1.0 * correct / (proposals + self.eps)
        recall    = 1.0 * correct / (total + self.eps)
        fscore    = 2.0 * precision * recall / (precision + recall + self.eps)
        print("precision: %f, recall: %f, fscore: %f" %
              (precision, recall, fscore))

        with open('log_test.txt', 'a') as log:
            log.write("Epoch %d: precision: %f, recall: %f, fscore: %f\n" %
                      (epoch, precision, recall, fscore))

    def detectFolder(self, path, saveMetrics=True):
        path = os.path.expanduser(path)

        IOU = ([], [], [], [], [], [], [], [], [], [],
               [], [], [], [], [], [], [], [], [], [])
        cm = np.zeros((20, 20), dtype=int)
        fileList = os.listdir(path)
        imgList = [path + '/' +
                   s for s in fileList if '.jpg' in s]

        if imgList == []:
            print ("Erro, Empty list")
            return ([-1], [-1], [-1])

        imgList = np.sort(imgList)
        width, height = self.init_width, self.init_height

        detectPath = path.replace("JPEGImages", "detect/")
        labelLogFolder = path.replace("JPEGImages", "predicted_labels")

        if os.path.exists(labelLogFolder) is not True:
            os.makedirs(labelLogFolder)

        if os.path.exists(detectPath) is not True:
            os.makedirs(detectPath)

        for img_url in imgList:

            gt_path = img_url.replace(
                "JPEGImages", "labels").replace("jpg", "txt")
            predicted_path = img_url.replace(
                "JPEGImages", "predicted_labels").replace("jpg", "txt")

            img    = Image.open(img_url).convert('RGB')
            sized  = img.resize((width, height))
            boxes  = ut.do_detect(self.model, sized, 0.5, 0.4, self.use_cuda)
            splitt = str(img_url).split('/')

            if img_url is img_url.replace("JPEGImages", "detect"):
                print ("Path in Unexpected format")
                return

            detections = []
            predictedLabels = []

            for i, box in enumerate(boxes):

                predictClass = box[6]

                xMin = (box[0] - box[2] / 2.0) * width
                yMin = (box[1] - box[3] / 2.0) * height
                xMax = (box[0] + box[2] / 2.0) * width
                yMax = (box[1] + box[3] / 2.0) * height
                bbox = np.array([[xMin, yMin], [xMax, yMax]])

                detections.append((predictClass, bbox))
                predictedLabels.append(
                    (predictClass, box[0], box[1], box[2], box[3]))

            if len(boxes) > 0:
                #print (detectPath + splitt[-1])
                ut.plot_boxes(img, boxes, detectPath +
                              splitt[-1], self.class_names)
                with open(predicted_path, 'w') as f:
                    for predict in predictedLabels:
                        f.write(' '.join(str(e) + '' for e in predict) + '\n')

            if not saveMetrics:
                continue

            truth = []
            with open(gt_path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):  # Each line is a Ground Truth
                    results = [float(key) for key in line.split()]
                    if len(results) != 0:
                        realClass = int(results[0])
                        gt_box = dt.VOCtoBBox(results, width, height)
                        truth.append((realClass, gt_box))

            # 1) Correct Class at correct place
            t_check = np.zeros(len(truth), dtype=int)
            d_check = np.zeros(len(detections), dtype=int)
            for i, (gt_class, gt_bbox) in enumerate(truth):
                if t_check[i] == 0:
                    for j, (det_class, det_bbox) in enumerate(detections):
                        if d_check[j] == 0:
                            new_IOU = dt.getIOU(gt_bbox, det_bbox)
                            #print(new_IOU, gt_class, det_class)
                            if new_IOU > 0.5 and gt_class is det_class:
                                cm[gt_class][det_class] += 1
                                t_check[i] = 1
                                d_check[j] = 1
                                # print gt_class
                                IOU[gt_class].append(new_IOU)

            # 2) Incorrect Class at correct place
            for i, (gt_class, gt_bbox) in enumerate(truth):
                if t_check[i] == 0:
                    for j, (det_class, det_bbox) in enumerate(detections):
                        if d_check[j] == 0:
                            new_IOU = dt.getIOU(gt_bbox, det_bbox)
                            if new_IOU > 0.5 and gt_class is not det_class:
                                cm[gt_class][det_class] += 1
                                t_check[i] = 1
                                d_check[j] = 1

            # 3) Truth not find
            for i in range(len(t_check)):
                if t_check[i] == 0:
                    (gt_class, _) = truth[i]
                    cm[gt_class][19] += 1
                    t_check[i] = 1
                    #print ("Error 3: " + img_url)

            # 4) Incorrect class at incorrect place
            for j in range(len(detections)):
                if d_check[j] == 0:
                    (det_class, _) = detections[j]
                    d_check[j] = 1
                    cm[19][det_class] += 1
                    #print ("Error 4: " + img_url)

            if os.path.exists(detectPath) is not True:
                os.makedirs(detectPath)

            with open(predicted_path, 'w') as f:
                for predict in predictedLabels:
                    f.write(' '.join(str(e) + '' for e in predict) + '\n')

        logUrl = "Log/"
        if os.path.exists(logUrl) is not True:
            os.makedirs(logUrl)

        timeStamp = (datetime.now())

        logName = path.split('/')[-2] + '_'
        logName += '_'.join(str(x) for x in (timeStamp.year,
                                             timeStamp.month, timeStamp.day, timeStamp.minute))
        logName += '.txt'

        timeStamp = str(timeStamp)

        print ("Creating Log : {}".format(logUrl + logName))
        with open(logUrl + logName, 'w') as f:
            IOU_class = []
            for i in range(20):
                IOU_class.append(
                    (sum(IOU[i]) / len(IOU[i])) if len(IOU[i]) > 0 else (-1))
            #print (IOU_class)

            f.write("YoloV2 Log at " + timeStamp + '\n')
            f.write("Using: " + self.weightfile + " weight file\n")
            f.write("At: " + path + " images\n\n")
            for i in range(20):
                f.write("{:02}: {}   IOU: ".format(
                    i, self.class_names[i]) + '{0:.3f}'.format(IOU_class[i]) + '\n')
            f.write("\nConfusion matrix\n\\\   ")

            for i in range(20):
                f.write("{:02}  ".format(i))
            f.write('\n')
            cmMatrix = ""
            for i in range(20):
                cmLine = "{:02}| ".format(i)
                for j in range(20):
                    cmLine += ("{:3d} ".format(cm[i][j]))
                cmMatrix += cmLine + '\n'
            f.write(cmMatrix)

            f.write("\n")
            f.write("Acurracy {0:.3f}".format(cm.trace() / float(cm.sum())))

        return IOU

    def detect(self, imgfile, calculateIOU=False):
        IOU = []
        count = np.zeros(20)
        #namesfile = '../Config/cc.names'

        img = Image.open(imgfile).convert('RGB')
        sized = img.resize((self.init_width, self.init_height))
        boxes = ut.do_detect(self.model, sized, 0.5, 0.4, self.use_cuda)

        width = self.init_width  # img.width
        height = self.init_height  # img.height

        #class_names = ut.load_class_names(namesfile)
        splitt = str(imgfile).split('/')
        detectPath = ''

        if imgfile is imgfile.replace("JPEGImages", "detect"):
            detectPath = imgfile
            detectPath = detectPath[:detectPath.rfind('/')] + "_detected/"
            calculateIOU = False
        else:
            detectPath = imgfile.replace("JPEGImages", "detect")
            detectPath = detectPath[:detectPath.rfind('/') + 1]

        for i in range(len(boxes)):
            count[boxes[i][6]] += 1

            xMin = (boxes[i][0] - boxes[i][2] / 2.0) * width
            yMin = (boxes[i][1] - boxes[i][3] / 2.0) * height
            xMax = (boxes[i][0] + boxes[i][2] / 2.0) * width
            yMax = (boxes[i][1] + boxes[i][3] / 2.0) * height
            bbox = np.array([[xMin, yMin], [xMax, yMax]])

            if calculateIOU:
                gt_path = imgfile.replace(
                    "JPEGImages", "labels").replace("jpg", "txt")
                with open(gt_path, 'r') as f:
                    box_IOU = []
                    lines = f.readlines()
                    for line in lines:
                        results = [float(i) for i in line.split()]
                        gt_box = dt.VOCtoBBox(results, width, height)
                        box_IOU.append(dt.getIOU(gt_box, bbox))
                        print ('**' * 60, IOU)
                IOU.append(max(box_IOU))

        if os.path.exists(detectPath) is not True:
            os.makedirs(detectPath)
        ut.plot_boxes(img, boxes, detectPath + splitt[-1], self.class_names)

        return count, IOU

    def imPlot(self, img, boxes):
        #namesfile = '../Config/cc.names'
        #class_names = ut.load_class_names(namesfile)
        width = self.init_width
        height = self.init_height
        color = (0, 255, 255)
        for i in range(len(boxes)):
            box = boxes[i]
            x1 = (box[0] - box[2] / 2.0) * width
            y1 = (box[1] - box[3] / 2.0) * height
            x2 = (box[0] + box[2] / 2.0) * width
            y2 = (box[1] + box[3] / 2.0) * height

            if len(box) >= 7 and self.class_names:
                cls_id = box[6]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, self.class_names[cls_id], (int(
                    x1), int(y1)), font, 2, color, thickness=2)
            cv2.rectangle(img, (int(x1), int(y1)),
                          (int(x2), int(y2)), color, thickness=2)
        return img

    def detectCam(self, index=0):
        cam = cv2.VideoCapture(index)

        if not cam.isOpened():
            print ("Video not found or Opencv without ffmpeg")
            return
        ret, src = cam.read()
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        while type(src) is not type(None):

            src = cv2.resize(src, (self.init_width, self.init_height))
            boxes = ut.do_detect(self.model, src, 0.5, 0.4, self.use_cuda)

            src = self.imPlot(src, boxes)
            src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
            cv2.imshow("Detect", src)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cam.release()
                cv2.destroyAllWindows()
                break

            ret, src = cam.read()
            src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

    def __initConfigProvider(self):
        self.cp = config.ConfigProvider(
            self.datacfg, self.cfgfile, self.weightfile)

        self.batch_size     = self.cp.batch_size
        self.nsamples       = self.cp.trainSample
        self.testDir        = self.cp.getTestUrl()
        self.trainDir       = self.cp.getTrainUrl()
        self.momentum       = self.cp.momentum
        self.lr             = self.cp.learning_rate
        self.decay          = self.cp.decay
        self.steps          = self.cp.steps
        self.scales         = self.cp.scales
        self.backupDir      = self.cp.backupDir
        self.save_interval  = self.cp.save_interval
        self.conf_thresh    = self.cp.conf_thresh
        self.nms_thresh     = self.cp.nms_thresh
        self.iou_thresh     = self.cp.iou_thresh
        self.eps            = self.cp.eps
        self.use_cuda       = self.cp.USE_CUDA

    def __init_test_loader(self):

        self.test_loader = torch.utils.data.DataLoader(
            dataset.listDataset(self.testDir, shape=(
                self.init_width, self.init_height), shuffle=False,
                transform=transforms.Compose([transforms.ToTensor(), ]),
                train=False, seen=self.model.seen,
                batch_size=self.batch_size,
                num_workers=self.num_workers),
            batch_size=self.batch_size, shuffle=False, **self.kwargs)


if __name__ == '__main__':
    init = 40
    weight = "Backup/{:06d}.weights".format(init)
    yolocfg = 'Config/yolo-voc.cfg'
    namesfile = 'Config/table.names'
    yolo = Yolo('Config/restaurant_table.csv', yolocfg, weight, namesfile)

    yolo.save_interval = 5

    # for i in tqdm(range(init, init + 1)):
    #yolo.train(i, fixLR=0.00001)
    # yolo.test(i)

    alfa = yolo.detectFolder('~/.Datasets/RM1/JPEGImages')
# =============================================================================
    #beta = yolo.detectCam(0)
# =============================================================================
