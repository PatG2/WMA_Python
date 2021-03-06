import cv2
import numpy as np
import numpy as np
import tensorflow as tf

#wczytanie moedlu
model = tf.keras.models.load_model("C:/Users/patry/m2.h5")
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

font = cv2.FONT_HERSHEY_SIMPLEX

#Wczytanie orginalnego obrazu:
image_small= cv2.imread('1.jpg.JPEG')
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
    
sift = cv2.SIFT_create() #SIFT w duzym uogólnieniu szuka kluczowych puntków na obrazie
bf = cv2.BFMatcher()

good_park = []
good_park_zas = []
good_stacja = []
good_przejscie_piesi = []
good_przejscie_rowery = []


#windowname = "ROI"

for contour in good_contours:
    x,y,w,h = cv2.boundingRect(contour)
    ROI = image_small[y:y+h, x:x+w]
    ROI_gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    ROI_mask = cv2.adaptiveThreshold(ROI_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 0)
    #cv2.imshow(windowname, ROI_mask)
    #windowname = windowname + '1'
    
    
    resized_up = cv2.resize(ROI, (150,150), interpolation= cv2.INTER_AREA)
    input_arr = np.array([resized_up])
    predictions = model.predict(input_arr)
    nr=[]
    for pred in predictions:
        n=0
        for value in pred:
            if value==max(pred):
                nr.append(n);
            n+=1
    print(nr)
    
    
    #Szukanie kluczowych punktów i deksryptorów z użyciem Scale-Invariant Feature Transform
    cv2.putText(image_small,str(nr),(x-50,y), font, 0.5,(0,255,0),2,cv2.LINE_AA)
    kp1, des1 = sift.detectAndCompute(ROI_mask, None)
    kp_park, des_park = sift.detectAndCompute(park_mask, None)
    kp_park_zas, des_park_zas = sift.detectAndCompute(park_zas_mask, None)
    kp_stacja, des_stacja = sift.detectAndCompute(stacja_mask, None)
    kp_przejscie_piesi, des_przejscie_piesi = sift.detectAndCompute(przejscie_piesi_mask, None)
    kp_przejscie_rowery, des_przejscie_rowery = sift.detectAndCompute(przejscie_rowery_mask, None)

    #porównanie kluczowych punktów miedzy szablonem a zdjeciem
    matches_park = bf.knnMatch(des1,des_park,k=2)
    matches_park_zas = bf.knnMatch(des1,des_park_zas,k=2)
    matches_stacja = bf.knnMatch(des1,des_stacja,k=2)
    matches_przejscie_piesi = bf.knnMatch(des1,des_przejscie_piesi,k=2)
    matches_przejscie_rowery = bf.knnMatch(des1,des_przejscie_rowery,k=2)

    #Ratio test
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
        cv2.putText(image_small,'Parking',(x,y), font, 0.5,(0,0,255),2,cv2.LINE_AA)

    if len(good_park_zas) > len(good_park) and len(good_park_zas) > len(good_stacja) and len(good_park_zas) > len(good_przejscie_piesi) and len(good_park_zas) > len(good_przejscie_rowery):
        print("Jest to znak parkingu zastrzezonego!") 
        cv2.putText(image_small,'Parking zastrzezony',(x,y), font, 0.5,(0,0,255),2,cv2.LINE_AA)
        
    if len(good_stacja) > len(good_park_zas) and len(good_stacja) > len(good_park) and len(good_stacja) > len(good_przejscie_piesi) and len(good_stacja) > len(good_przejscie_rowery):
        print("Jest to znak stacji paliw!")
        cv2.putText(image_small,'Stacja paliw',(x,y), font, 0.5,(0,0,255),2,cv2.LINE_AA)
        
    if len(good_przejscie_piesi) > len(good_park_zas) and len(good_przejscie_piesi) > len(good_stacja) and len(good_przejscie_piesi) > len(good_park) and len(good_przejscie_piesi) > len(good_przejscie_rowery):
        print("Jest to znak przejscia dla pieszych!")
        cv2.putText(image_small,'Przejscie dla pieszych',(x,y), font, 0.5,(0,0,255),2,cv2.LINE_AA)
        
    if len(good_przejscie_rowery) > len(good_park_zas) and len(good_przejscie_rowery) > len(good_stacja) and len(good_przejscie_rowery) > len(good_przejscie_piesi) and len(good_przejscie_rowery) > len(good_park):
        print("Jest to znak przejazdu dla rowerow!")
        cv2.putText(image_small,'Przejazd dla rowerow',(x,y), font, 0.5,(0,0,255),2,cv2.LINE_AA)
   
    if len(good_park)==0 and len(good_park_zas) ==0 and len(good_stacja) ==0 and len(good_przejscie_piesi) == 0:    
        print("Nie udalo sie rozpoznac znaku!")
        cv2.putText(image_small,'Nie rozpoznano znaku',(x,y), font, 0.5,(0,0,255),2,cv2.LINE_AA)
      
cv2.imshow("Rezultat", image_small)  
cv2.waitKey()
cv2.destroyAllWindows()