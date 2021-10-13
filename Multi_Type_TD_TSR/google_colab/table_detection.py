import cv2
# from google.colab.patches import cv2_imshow
import numpy as np
from matplotlib import pyplot as plt

def plot_prediction(img, predictor):
    
    outputs = predictor(img)

    # Blue color in BGR 
    color = (255, 0, 0) 
  
    # Line thickness of 2 px 
    thickness = 2

    for x1, y1, x2, y2 in outputs["instances"].get_fields()["pred_boxes"].tensor.to("cpu").numpy():
        # Start coordinate 
        # represents the top left corner of rectangle 
        start_point = (x1, y1) 
  
        # Ending coordinate
        # represents the bottom right corner of rectangle 
        end_point = (x2, y2) 
  
        # Using cv2.rectangle() method 
        # Draw a rectangle with blue line borders of thickness of 2 px 
        img = cv2.rectangle(np.array(img, copy=True), start_point, end_point, color, thickness)

    # Displaying the image
    print("TABLE DETECTION:")  
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # source: https://stackoverflow.com/questions/38598118/difference-between-plt-show-and-cv2-imshow
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def make_prediction(img, predictor):
    
    #img = cv2.imread(img_path)
    outputs = predictor(img)

    table_list = []
    table_coords = []
    num_plots = len(outputs["instances"].get_fields()["pred_boxes"].tensor.to("cpu").numpy())
    print("Number of tables on this page: {0}".format(num_plots))

    # source: https://stackoverflow.com/questions/41210823/using-plt-imshow-to-display-multiple-images
    plt.figure(figsize=(50, 50))
    figure, axis = plt.subplots(num_plots, 1)

    for i, box in enumerate(outputs["instances"].get_fields()["pred_boxes"].tensor.to("cpu").numpy()):
        x1, y1, x2, y2 = box
        table_list.append(np.array(img[int(y1):int(y2), int(x1):int(x2)], copy=True))
        table_coords.append([int(x1),int(y1),int(x2-x1),int(y2-y1)])
        axis[i].imshow(cv2.cvtColor(img[int(y1):int(y2), int(x1):int(x2)], cv2.COLOR_BGR2RGB))
        print()
        
    return table_list, table_coords
