from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import logging
import cv2

MODEL_FILE_PATH = 'models\\custom_trained\\mb1-ssd-Epoch-211-Loss-2.275852680206299.pth'
LABEL_FILE_PATH = 'labels\\labels.txt'
LOG_FILE_PATH = 'logs\\vehicle_plate_detector_logger.log'
TEST_FILE_PATH = 'dataset\\VOC2007\\ImageSets\\Main\\test.txt'
RUNTIME_DEVICE_TYPE = 'cpu'

logging.basicConfig(filename=LOG_FILE_PATH, filemode='a', level=logging.INFO,\
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.info('Vehicle Plate Inference Engine')

class VehiclePlateDetector():

    def __init__(self, net_type, model_path, label_path, device):
        
        try:
            self.net_type = net_type
            self.model_path = model_path
            self.label_path = label_path
            self.class_names = [name.strip() for name in open(self.label_path).readlines()]
            self.timer = Timer()

            if self.net_type == 'vgg16-ssd':
                self.net = create_vgg_ssd(len(self.class_names ), is_test=True, device=device)
            elif self.net_type == 'mb1-ssd':
                self.net = create_mobilenetv1_ssd(len(self.class_names ), is_test=True, device=device)
            elif self.net_type == 'mb1-ssd-lite':
                self.net = create_mobilenetv1_ssd_lite(len(self.class_names ), is_test=True, device=device)
            elif self.net_type == 'mb2-ssd-lite':
                self.net = create_mobilenetv2_ssd_lite(len(self.class_names ), is_test=True, device=device)
            elif self.net_type == 'sq-ssd-lite':
                self.net = create_squeezenet_ssd_lite(len(self.class_names ), is_test=True, device=device)
            else:
                logging.info(f"Init from base net {args.base_net}")
                print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")

            self.net.load(self.model_path)

            if self.net_type == 'vgg16-ssd':
                self.predictor = create_vgg_ssd_predictor(self.net, candidate_size=200, device=device)
            elif self.net_type == 'mb1-ssd':
                self.predictor = create_mobilenetv1_ssd_predictor(self.net, candidate_size=200, device=device)
            elif self.net_type == 'mb1-ssd-lite':
                self.predictor = create_mobilenetv1_ssd_lite_predictor(self.net, candidate_size=200, device=device)
            elif self.net_type == 'mb2-ssd-lite':
                self.predictor = create_mobilenetv2_ssd_lite_predictor(self.net, candidate_size=200, device=device)
            elif self.net_type == 'sq-ssd-lite':
                self.predictor = create_squeezenet_ssd_lite_predictor(self.net, candidate_size=200, device=device)
            else:
                self.predictor = create_vgg_ssd_predictor(self.net, candidate_size=200, device=device)
        except Exception as e:
            logging.error("Caught an exception in VehiclePlateDetector constructor", exc_info=True)
            print("Caught an exception in VehiclePlateDetector constructor", e)

    def detect_objects(self, image):
        
        try:
            self.timer.start("Load Image")
            orig_image = cv2.imread(image)
            load_end_time = self.timer.end("Load Image")
            print("Load Image: {:4f} seconds.".format(load_end_time))
            logging.info(f"Load Image seconds {load_end_time}")

            self.timer.start("Predict")
            boxes, labels, probs = self.predictor.predict(orig_image, 10, 0.4)
            predict_end_time = self.timer.end("Predict")
            print("Prediction: {:4f} seconds.".format(predict_end_time))
            logging.info(f"Prediction seconds {predict_end_time}")
            print("---------------------------------------------------------------------")

            for i in range(boxes.size(0)):
                box = boxes[i, :]
                c1 = int(box[0].numpy())
                c2 = int(box[1].numpy())
                c3 = int(box[2].numpy())
                c4 = int(box[3].numpy())
                cv2.rectangle(orig_image, (c1, c2), (c3, c4), (255, 255, 0), 4)
                label = f"{self.class_names [labels[i]]}: {probs[i]:.2f}"
                print("Vehicle Plate Detector Prediction ", label)
                logging.info(f"Vehicle Plate Detector Prediction  {label}")
                cv2.putText(orig_image, label, (c1-40, c2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                file_save_path = 'results\\output_' + str(count) + '.jpg' 
                cv2.imwrite(file_save_path, orig_image)
            return predict_end_time
        except Exception as e:
            logging.error("Caught an exception in detect_objects", exc_info=True)
            print("Caught an exception in detect_objects", e)
            
if __name__ == '__main__':
    
    try:
        detector = VehiclePlateDetector('mb1-ssd', MODEL_FILE_PATH, LABEL_FILE_PATH, RUNTIME_DEVICE_TYPE)
        file1 = open(TEST_FILE_PATH, 'r')
        lines = file1.readlines()
        count = 1
        batch_count = 0
        batch_inference_time = 0
        batch_inference_time_container = {}

        for file_name in range(len(lines)):
            file_path = 'dataset\\VOC2007\\JPEGImages\\' + str(lines[file_name].strip('\n')) + '.jpg'
            print("---------------------------------------------------------------------")
            print("Process image - ", count)
            logging.info(f"Process image {count}")
            predict_end_time = detector.detect_objects(file_path)
            if count % 30 == 0 or count == len(lines):
                batch_no = "batch_" + str(batch_count)
                batch_inference_time_container[batch_no] = batch_inference_time
                batch_inference_time = 0
                batch_count += 1
            else:
                batch_inference_time += predict_end_time

            count += 1

        print("Batch wise Prediction Time :- ", batch_inference_time_container)
        logging.info(f"Batch wise Prediction Time :-  {batch_inference_time_container}")
        print("------------------------------------------------------------------------")
    except Exception as e:
        logging.error("Caught an exception in main function", exc_info=True)
        print("Caught an exception in main function", e)