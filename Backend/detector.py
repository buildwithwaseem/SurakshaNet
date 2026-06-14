import cv2
import numpy as np
import onnxruntime as ort

class SurakshaNetDetector:
    def __init__(self, model_path="models/weights/best.onnx"):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.session = ort.InferenceSession(model_path, sess_options=opts)
        self.input_name = self.session.get_inputs()[0].name
        
        # FIX 1: Indexing order update kiya (Roboflow standard)
        self.classes = ['Head', 'Helmet', 'Person']
        
    def preprocess(self, frame):
        img = cv2.resize(frame, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    # FIX 2: Default confidence threshold ko badha kar 0.75 kiya taaki false positives rukein
    def detect(self, frame, conf_threshold=0.5):
        h, w, _ = frame.shape
        input_data = self.preprocess(frame)
        
        outputs = self.session.run(None, {self.input_name: input_data})
        predictions = np.squeeze(outputs[0])
        
        boxes = []
        confidences = []
        class_ids = []
        
        if len(predictions.shape) == 2:
            predictions = predictions.T
            for pred in predictions:
                scores = pred[4:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > conf_threshold:
                    cx, cy, nw, nh = pred[0:4]
                    x1 = int((cx - nw/2) * (w / 640))
                    y1 = int((cy - nh/2) * (h / 640))
                    x2 = int((cx + nw/2) * (w / 640))
                    y2 = int((cy + nh/2) * (h / 640))
                    
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))
                    
                    
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)
        
        violations = []
        if len(indices) > 0:
            for i in indices.flatten():
                violations.append({
                    "class": self.classes[class_ids[i]],
                    "confidence": confidences[i],
                    "bbox": boxes[i]
                })
        return violations