import numpy as np
import cv2
import pickle
from matplotlib import pyplot as plt
from pyzbar.pyzbar import decode
from pyzbar.pyzbar import ZBarSymbol

# Show image
def show(img, cmap='gray'):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10,16))
    ax.imshow(img, cmap)

# Capture image
def capImage():
    cameraCapture = cv2.VideoCapture(0) # open usb camera
    cameraCapture.set(15, -6.0); # Modified
    cameraCapture.set(3,1920)
    cameraCapture.set(4,1080)
    success, img = cameraCapture.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if success == False:
        print("No image captured.")
    else:
        show(img, None)
    cameraCapture.release() 
    return img

# Detect wheels 
def detectWheels(img):
    # Process image
    (_, thresh) = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.dilate(thresh, kernel, iterations = 1)

    qr = decode(closed, symbols=[ZBarSymbol.QRCODE])

    wheel_mid_points = np.zeros((4,2))
    wheel_orient = np.zeros((4,2))
    for i in range(len(qr)):    
        # Find middle point of qrcode
        points = qr[i].polygon
        pt_mid = np.mean(points, axis=0)
        pt_mid = np.int32(pt_mid)
        wheel_mid_points[int(qr[i][0])][0] = (pt_mid[0]-1920/2)*30/131
        wheel_mid_points[int(qr[i][0])][1] = -(pt_mid[1]-1080/2)*30/131+70
            
        # Find orientation of qrcode
        vx = points[0][0] - points[1][0]
        vy = points[0][1] - points[1][1]
        wheel_orient[int(qr[i][0])] = [vx, vy]
    return wheel_mid_points, wheel_orient

# Detect control box
def detectCBox(img):
    # Preprocessing--------------------------------------------------------------------
    imgsub = img[:600,350:1350]
    imgflt = cv2.bilateralFilter(imgsub,9,75,75)
    # Pick object indices---------------------------------------------------------------
    inds = np.argwhere(imgflt>210)
    indf = np.fliplr(inds)
    # OBB------------------------------------------------------------------------------
    ca = np.cov(indf,y = None,rowvar = 0,bias = 0)
    v, vect = np.linalg.eig(ca)
    idx = v.argsort()[::-1]
    vect = vect[:,idx]
    tvect = np.transpose(vect)
    ar = np.dot(indf,np.linalg.inv(tvect))
    # get the minimum and maximum x and y 
    mina = np.min(ar,axis=0)
    maxa = np.max(ar,axis=0)
    diff = (maxa - mina)*0.5
    # the center is just half way between the min and max xy
    center = mina + diff
    # get the 4 corners by subtracting and adding half the bounding boxes height and width to the center
    #corners = np.array([center+[-diff[0],-diff[1]],center+[diff[0],-diff[1]],center+[diff[0],diff[1]],center+[-diff[0],diff[1]],center+[-diff[0],-diff[1]]])
    # use the the eigenvectors as a rotation matrix and
    # rotate the corners and the centerback
    cb_center = np.dot(center,tvect) 
    cb_center[0] = cb_center[0] + 350
    # Convert to real position
    cb_center[0] = (cb_center[0] - 1920/2)*30/112-5
    cb_center[1] = -(cb_center[1] - 1080/2)*30/112 + 70


    sum1 = 0 # Sum values along the vector 
    sum2 = 0
    print tvect[0]
    for i in inds:
        if tvect[0].dot(i-cb_center) < 0:
            sum1 = sum1 + img[i[0], i[1]]
        else:
            sum2 = sum2 + img[i[0], i[1]]            
    cb_orient = tvect[0] if sum1 > sum2 else -tvect[0]

    return cb_center, cb_orient

# Detect battery
def detectBat(img):
    # Preprocessing--------------------------------------------------------------------
    imgsub = img[600:,350:1350]
    imgflt = cv2.bilateralFilter(imgsub,9,75,75)
    # Pick object indices---------------------------------------------------------------
    inds = np.argwhere(imgflt>200)
    indf = np.fliplr(inds)
    # OBB------------------------------------------------------------------------------
    ca = np.cov(indf,y = None,rowvar = 0,bias = 0)
    v, vect = np.linalg.eig(ca)
    idx = v.argsort()[::-1]
    vect = vect[:,idx]
    tvect = np.transpose(vect)
    ar = np.dot(indf,np.linalg.inv(tvect))
    # get the minimum and maximum x and y 
    mina = np.min(ar,axis=0)
    maxa = np.max(ar,axis=0)
    diff = (maxa - mina)*0.5
    # the center is just half way between the min and max xy
    center = mina + diff
    # get the 4 corners by subtracting and adding half the bounding boxes height and width to the center
    #corners = np.array([center+[-diff[0],-diff[1]],center+[diff[0],-diff[1]],center+[diff[0],diff[1]],center+[-diff[0],diff[1]],center+[-diff[0],-diff[1]]])
    #bat_corners = np.dot(corners,tvect)
    bat_center = np.dot(center,tvect)
    # Coords convertion
    bat_center[0] = bat_center[0]+350
    bat_center[1] = bat_center[1]+600
    bat_center[0] = (bat_center[0] - 1920/2)*30/112-5
    bat_center[1] = -(bat_center[1] - 1080/2)*30/112 + 70

    
    sum1 = 0 # Sum values along the vector 
    sum2 = 0
    for i in inds:
        if tvect[0].dot(i-center) < 0:
            sum1 = sum1 + img[i[0], i[1]]
        else:
            sum2 = sum2 + img[i[0], i[1]]           
    bat_orient = tvect[0] if sum1 > sum2 else -tvect[0] 

    return bat_center, bat_orient
