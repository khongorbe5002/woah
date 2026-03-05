import cv2

img = cv2.imread('sample.jpg')

if img is None:
    print("Error: could not find the image")
    
else:
    print("image loaded")
    
    cv2.imshow('open cv image test', img)
    
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    

    
