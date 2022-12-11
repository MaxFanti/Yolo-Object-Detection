import cv2 as cv
import numpy as np
import time
import mss
import multiprocessing
from multiprocessing import Pipe
import torch

display_time = 1
fps = 0
start_time = time.time()
monitor = {"top": 0, "left": 0, "width": 800, "height": 640}
sct = mss.mss()


def Grab_screen(p_input):
    while (True):
        screenshot = np.array(sct.grab(monitor))
        screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2BGR)
        p_input.send(screenshot)


def Yolov7_render(p_output, p_input2):

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    while (True):
        image = p_output.recv()
        result = model(image)
        out = np.squeeze(result.render())
        p_input2.send(out)


def Show_image(p_output2):
    global start_time, fps
    while (True):
        image = p_output2.recv()
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.imshow('Object Detection', image)
        fps += 1
        TIME = time.time() - start_time
        if (TIME) >= display_time:
            print("FPS: ", fps / (TIME))
            fps = 0
            start_time = time.time()
        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break


if __name__ == "__main__":
    # Pipes
    p_output, p_input = Pipe()
    p_output2, p_input2 = Pipe()
    # creating new processes
    p1 = multiprocessing.Process(target=Grab_screen, args=(p_input,))
    p2 = multiprocessing.Process(
        target=Yolov7_render, args=(p_output, p_input2))
    p3 = multiprocessing.Process(target=Show_image, args=(p_output2,))

    # starting our processes
    p1.start()
    p2.start()
    p3.start()
