import numpy as np
import cv2 as cv
import onnx
import onnxruntime as ort


print(ort.get_device())
# Load the ONNX model
model = onnx.load("mynet.onnx")
print(type(model))
# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph

print(onnx.helper.printable_graph(model.graph))
ort_session = ort.InferenceSession("mynet.onnx",providers=['CPUExecutionProvider'])
input_size = (28,28)


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    upper_point = (frame.shape[1]//2 + 100, frame.shape[0]//2 + 100)
    lower_point = (frame.shape[1]//2 - 100, frame.shape[0]//2 - 100)
    
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    roi = gray[frame.shape[0]//2 - 100:frame.shape[0]//2 + 100, frame.shape[1]//2 - 100:frame.shape[1]//2 + 100]
    roi_norm = (roi - np.mean(roi)) / np.std(roi)
    roi_resized = cv.resize(roi_norm, input_size)
    # Display the resulting frame
    
    cv.imshow('digit', roi_resized)
    cv.rectangle(frame, lower_point, upper_point, (0,0,255), 2, cv.FILLED)
    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('i'):
        input_tensor = roi_resized.reshape((1, 1, input_size[0], input_size[1])).astype(np.float32)

        # Inference
        ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
        output = ort_session.run(None, ort_inputs)[0]
        pred = int(np.argmax(output, axis=1))
        print("Digit found: {}".format(pred))


# When everything done, release the capture
cap.release()
cv.destroyAllWindows()