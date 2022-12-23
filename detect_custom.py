import cv2
import numpy 
import time
import mss
import multiprocessing
from multiprocessing import Pipe
import torch
from models.yolo import Model
from utils.torch_utils import select_device

monitor = {"top": 0, "left": 0, "width": 800, "height": 600}
sct = mss.mss()
display_time = 1
fps = 0
start_time = time.time()


def GetFpsCount():
    global fps, start_time
    fps += 1
    TIME = time.time() - start_time
    if (TIME) >= display_time:
        print("FPS: ", fps / (TIME))
        fps = 0
        start_time = time.time()

def GrubScreen(p_input):
    while True:
        sct_img = sct.grab(monitor)
        sct_img = numpy.array(sct_img)
        sct_img = cv2.cvtColor(sct_img, cv2.COLOR_RGB2BGR)
        p_input.send(sct_img)
        
def Yolov7_render(p_output, p_input2):
    path_or_model = './best.pt'  # path to model
    model = torch.load(path_or_model, map_location=torch.device('cpu')) if isinstance(
        path_or_model, str) else path_or_model  # load checkpoint
    if isinstance(model, dict):
        model = model['ema' if model.get('ema') else 'model']  # load model
    hub_model = Model(model.yaml).to(next(model.parameters()).device)
    hub_model.load_state_dict(model.float().state_dict())  # load state_dict
    hub_model.names = model.names  # class names
    hub_model = hub_model.autoshape()
    device = select_device('0' if torch.cuda.is_available() else 'cpu')
    model = hub_model.to(device)
    while (True):
        image = p_output.recv()
        result = model(image)
        out = numpy.squeeze(result.render())
        p_input2.send(out)

def ShowImage(p_output2):
    while True:  
        img = p_output2.recv()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Screenshot!", img)
        GetFpsCount()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



if __name__ == "__main__":

    p_output, p_input = Pipe()
    p_output2, p_input2 = Pipe()

    p1 = multiprocessing.Process(target=GrubScreen, args=(p_input,))
    p2 = multiprocessing.Process(
        target=Yolov7_render, args=(p_output, p_input2))
    p3 = multiprocessing.Process(target=ShowImage, args=(p_output2,))


    p1.start()
    p2.start()
    p3.start()