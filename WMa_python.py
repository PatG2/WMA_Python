import cv2
import numpy as np
import math

 
#wczytanie bazowych zdjec
park_shape = cv2.imread('Parking_shape.PNG')
park_zas_shape = cv2.imread('Parking_zastrzezone_shape.PNG')
stacja_shape = cv2.imread('Stacja_paliwowa_shape.PNG')
przejscie_piesi_shape = cv2.imread('przejscie_dla_pieszych_shape.PNG')
przejscie_rowery_shape = cv2.imread('przejazd_dla_rowerow_shape.PNG')

white_lower_shape = np.array([0,0,0])
white_upper_shape = np.array([0,0,255])

park_hsv = cv2.cvtColor(park_shape, cv2.COLOR_BGR2HSV)
park_zas_hsv = cv2.cvtColor(park_zas_shape, cv2.COLOR_BGR2HSV)
stacja_hsv = cv2.cvtColor(stacja_shape, cv2.COLOR_BGR2HSV)
przejscie_piesi_hsv = cv2.cvtColor(przejscie_piesi_shape, cv2.COLOR_BGR2HSV)
przejscie_rowery_hsv = cv2.cvtColor(przejscie_rowery_shape, cv2.COLOR_BGR2HSV)

park_mask = cv2.inRange(park_hsv, white_lower_shape, white_upper_shape)
park_zas_mask = cv2.inRange(park_zas_hsv, white_lower_shape, white_upper_shape)
stacja_mask = cv2.inRange(stacja_hsv, white_lower_shape, white_upper_shape)
przejscie_piesi_mask = cv2.inRange(przejscie_piesi_hsv, white_lower_shape, white_upper_shape)
przejscie_rowery_mask = cv2.inRange(przejscie_rowery_hsv, white_lower_shape, white_upper_shape)

park_moments = cv2.moments(park_mask)
park_zas_moments = cv2.moments(park_zas_mask)
stacja_moments = cv2.moments(stacja_mask)
przejscie_piesi_moments = cv2.moments(przejscie_piesi_mask)
przejscie_rowery_moments = cv2.moments(przejscie_rowery_mask)

park_hu = cv2.HuMoments(park_moments)
park_zas_hu = cv2.HuMoments(park_zas_moments)
stacja_hu = cv2.HuMoments(stacja_moments)
przejscie_piesi_hu = cv2.HuMoments(przejscie_piesi_moments)
przejscie_rowery_hu = cv2.HuMoments(przejscie_rowery_moments)

#for i in range(0,7):
#    park_hu[i] = -1* math.copysign(1.0, park_hu[i]) * math.log10(abs(park_hu[i]))
#    park_zas_hu[i] = -1* math.copysign(1.0, park_zas_hu[i]) * math.log10(abs(park_zas_hu[i]))
#    stacja_hu[i] = -1* math.copysign(1.0, stacja_hu[i]) * math.log10(abs(stacja_hu[i]))
#    przejscie_piesi_hu[i] = -1* math.copysign(1.0, przejscie_piesi_hu[i]) * math.log10(abs(przejscie_piesi_hu[i]))
#    przejscie_rowery_hu[i] = -1* math.copysign(1.0, przejscie_rowery_hu[i]) * math.log10(abs(przejscie_rowery_hu[i]))
    
#print(park_hu, "park")
#print(park_zas_hu, "park zas")
#print(stacja_hu, "stacja")
#print(przejscie_piesi_hu, "piesi")
#print(przejscie_rowery_hu, "rowey")

#Wczytanie orginalnego obrazu:
image_small= cv2.imread('11.jpg.JPEG')
image_small = cv2.pyrDown(image_small)

#filtr medianowy
median = cv2.medianBlur(image_small,3)

#Zmiana przestrzeni barw hsv jest niewrażliwe na światło (zmiany w podaniu, natężeniu)
hsv_image = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)

#Definicja ograniczeń dla niebieskiego koloru w hsv
blue_lower = np.array([100,160,60])
blue_upper = np.array([120,255,180])

#maska nakładana na obraz w celu wykrycia elementów tylko z przedziału wartości zdefiniowanej dla ograniczeń koloru niebieskiego
#wynikiem jest obraz binarny
mask = cv2.inRange(hsv_image, blue_lower, blue_upper)

#elementy strukturalne do morfologicznych operacji
kernel1 = np.ones((2,2), np.uint8)
kernel2 = np.ones((9,9), np.uint8)

# różna ilość iteracji lepiej się sprawdza dla różnych obrazów 
mask_morph = cv2.erode(mask,kernel1,iterations=2)
mask_morph2 = cv2.dilate(mask_morph,kernel2,iterations=1)
#cv2.imshow("Highlighted3", mask_morph2)

#kontury na niebieskich elementach najzwyklejsze
contours, _ = cv2.findContours(mask_morph2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


good_contours = []
i = 0
#Sprawdzenie czy kontury są prostokątami/kwadratami
for contour in contours:
    approx = cv2.approxPolyDP(contour,0.035*cv2.arcLength(contour,True),True)
    if len(approx) == 4 and cv2.contourArea(approx) > 1200 and _[0,i,3] == -1:
        cv2.drawContours(image_small, [contour], 0, (0,255,0), 1) 
        good_contours.append(contour)
    cv2.drawContours(hsv_image, [contour], 0, (0,0,0),1)    
    i = i+1
    
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

good_park = []
good_park_zas = []
good_stacja = []
good_przejscie_piesi = []
good_przejscie_rowery = []


windowname = "ROI"
widow = "name"


for contour in good_contours:
    x,y,w,h = cv2.boundingRect(contour)
    ROI = image_small[y:y+h, x:x+w]
    ROI_gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    ROI_mask = cv2.adaptiveThreshold(ROI_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 45, 0)
    cv2.imshow(windowname, ROI_mask)
    windowname = windowname + '1'
    
    kp1, des1 = sift.detectAndCompute(ROI_mask, None)
    kp_park, des_park = sift.detectAndCompute(park_mask, None)
    kp_park_zas, des_park_zas = sift.detectAndCompute(park_zas_mask, None)
    kp_stacja, des_stacja = sift.detectAndCompute(stacja_mask, None)
    kp_przejscie_piesi, des_przejscie_piesi = sift.detectAndCompute(przejscie_piesi_mask, None)
    kp_przejscie_rowery, des_przejscie_rowery = sift.detectAndCompute(przejscie_rowery_mask, None)

    matches_park = bf.knnMatch(des1,des_park,k=2)
    matches_park_zas = bf.knnMatch(des1,des_park_zas,k=2)
    matches_stacja = bf.knnMatch(des1,des_stacja,k=2)
    matches_przejscie_piesi = bf.knnMatch(des1,des_przejscie_piesi,k=2)
    matches_przejscie_rowery = bf.knnMatch(des1,des_przejscie_rowery,k=2)

    for m,n in matches_park:
        if m.distance < 0.55*n.distance:
            good_park.append([m])

    for m,n in matches_park_zas:
        if m.distance < 0.55*n.distance:
            good_park_zas.append([m])
            
    for m,n in matches_stacja:
        if m.distance < 0.55*n.distance:
            good_stacja.append([m])

    for m,n in matches_przejscie_piesi:
        if m.distance < 0.55*n.distance:
            good_przejscie_piesi.append([m])

    for m,n in matches_przejscie_rowery:
        if m.distance < 0.55*n.distance:
            good_przejscie_rowery.append([m])
            
    if len(good_park) > len(good_park_zas) and len(good_park) > len(good_stacja) and len(good_park) > len(good_przejscie_piesi) and len(good_park) > len(good_przejscie_rowery):
        print("Jest to znak parkingu!")
    
    if len(good_park_zas) > len(good_park) and len(good_park_zas) > len(good_stacja) and len(good_park_zas) > len(good_przejscie_piesi) and len(good_park_zas) > len(good_przejscie_rowery):
        print("Jest to znak parkingu zastrzezonego!") 
        
    if len(good_stacja) > len(good_park_zas) and len(good_stacja) > len(good_park) and len(good_stacja) > len(good_przejscie_piesi) and len(good_stacja) > len(good_przejscie_rowery):
        print("Jest to znak stacji paliw!")
        
    if len(good_przejscie_piesi) > len(good_park_zas) and len(good_przejscie_piesi) > len(good_stacja) and len(good_przejscie_piesi) > len(good_park) and len(good_przejscie_piesi) > len(good_przejscie_rowery):
        print("Jest to znak przejscia dla pieszych!")
        
    if len(good_przejscie_rowery) > len(good_park_zas) and len(good_przejscie_rowery) > len(good_stacja) and len(good_przejscie_rowery) > len(good_przejscie_piesi) and len(good_przejscie_rowery) > len(good_park):
        print("Jest to znak przejazdu dla rowerow!")
    #ROI_moments = cv2.moments(ROI_mask)
    #ROI_hu = cv2.HuMoments(ROI_moments)
    #for i in range(0,7):  
     #   ROI_hu[i] = -1* math.copysign(1.0, ROI_hu[i]) * math.log10(abs(ROI_hu[i]))
    
   #print(ROI_hu)
   #match_park = cv2.matchShapes(ROI_mask, park_mask, 3, 1)
   #match_park_zas = cv2.matchShapes(ROI_mask, park_zas_mask, 3, 1)
   #match_stacja = cv2.matchShapes(ROI_mask, stacja_mask, 3, 1)
   #match_przejscie_piesi = cv2.matchShapes(ROI_mask, przejscie_piesi_mask, 3, 1)
   #match_przejscie_rowery = cv2.matchShapes(ROI_mask, przejscie_rowery_mask, 3, 1)
        
 

cv2.imshow("kontury222", stacja_mask) 
cv2.imshow("kontury22", park_mask) 
cv2.imshow("kontury2", park_zas_mask) 
cv2.imshow("kontur", przejscie_piesi_mask) 
cv2.imshow("kontu", przejscie_rowery_mask) 




        

#img3 = cv2.drawMatchesKnn(ROI_mask, kp1, stacja_mask, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("kontury", image_small)  
cv2.waitKey()
cv2.destroyAllWindows()