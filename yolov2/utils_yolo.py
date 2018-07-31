import numpy as np


def getIOU(bbox_true, bbox_pred):
    """      
        Return: 
            Intersection Over Union

        bbox_true (bbox): 
            BoundBox in format array([[xMin,  yMin], [xMax,  yMax]])

        bbox_pred (bbox): 
            BoundBox in format array([[xMin,  yMin], [xMax,  yMax]])


    """
    # Inter
    xMin = max(bbox_true[0][0], bbox_pred[0][0])
    yMin = max(bbox_true[0][1], bbox_pred[0][1])
    xMax = min(bbox_true[1][0], bbox_pred[1][0])
    yMax = min(bbox_true[1][1], bbox_pred[1][1])

    interA = 0
    if(xMax > xMin and yMax > yMin):
        interA = ((xMax - xMin) * (yMax - yMin))

    trueA = ((bbox_true[1][0] - bbox_true[0][0]) *
             (bbox_true[1][1] - bbox_true[0][1]))
    predA = ((bbox_pred[1][0] - bbox_pred[0][0]) *
             (bbox_pred[1][1] - bbox_pred[0][1]))
    # A U B = A + B - (A inter B)
    IOU = interA / (trueA + predA - interA)

    return IOU


def VOCtoBBox(vocLabel, imgW=1024, imgH=640):
    """
        vocLabel: 
             Classe; 
             absoluteX/imgWidth; absoluteY/imgHeight;
             absoluteWidth/imgWidth; absoluteHeight/imgHeigh 
    """
    xMean = vocLabel[1] * imgW
    yMean = vocLabel[2] * imgH

    deltaX = vocLabel[3] * imgW
    deltaY = vocLabel[4] * imgH

    dX = deltaX / 2
    dY = deltaY / 2
    xMin = xMean - dX
    xMax = xMean + dX
    yMin = yMean - dY
    yMax = yMean + dY

    return np.array([[xMin, yMin], [xMax, yMax]])
