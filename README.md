# Yolo | Capture Window Screen for Object Detection
You can capture only a part of the screen:
> monitor = {"top": 0, "left": 0, "width": 800, "height": 640}

# Multiprocessing
Performances can be improved by delegating the PNG file creation and Yolo detect objects to a specific worker.

    if __name__ == "__main__":
    # Pipes
      p_output, p_input = Pipe()
      p_output2, p_input2 = Pipe()
    # creating new processes
      p1 = multiprocessing.Process(target=Grab_screen, args=(p_input,))
      p2 = multiprocessing.Process(target=Yolov7_render, args=(p_output, p_input2))
      p3 = multiprocessing.Process(target=Show_image, args=(p_output2,))
    # starting our processes
      p1.start()
      p2.start()
      p3.start()
