from django.shortcuts import render
import gspread
import oauth2client
from oauth2client.service_account import ServiceAccountCredentials
import cv2
import numpy as np
import pytesseract

# scope = ['https://spreadsheets.google.com/feeds']
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']

creds = ServiceAccountCredentials.from_json_keyfile_name('IEEEWIE-9ea42a36af92.json', scope)
client = gspread.authorize(creds)

sheet = client.open('sample').sheet1

# Create your views here.

def base(request):
    return render(request,"base.html")


def data_view(request):

    MIN = 10
    img1 = cv2.imread('m.jpg', 0)
    text, text1 = '' ,''
    detection = False
    video = cv2.VideoCapture(0)
    while True:
        _, img2 = video.read()  # img2 = cv2.imread('m3.jpg', 0)
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:

            if m.distance < 0.7 * n.distance:
                good.append(m)
        if len(good) > MIN:

            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            img2 = cv2.polylines(img2, [np.int32(dst)], True, (255), 3, cv2.LINE_AA)
            print("*********CARD MATCHED*********")
        else:
            print("Not enough matches are found - %d/%d" %
                  (len(good), MIN))
            matchesMask = None
            break
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=None,
                           matchesMask=matchesMask,
                           flags=2)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        # plt.imshow(img3, 'gray'), plt.show()
        # print(dst)


        try:
            approx = np.int32(dst)
        except:
            data = {"text1 ": text, "text2": text1, 'detection': detection}
            return render(request, "data_view.html", data)

        # print(approx)

        # approxx=mapp(approx) #find endpoints of the sheet

        # pts=np.float32([[0,0],[800,0],[800,800],[0,800]])  #map to 800*800 target window

        # op=cv2.getPerspectiveTransform(approxx,pts)  #get the top or bird eye view effect
        # dst=cv2.warpPerspective(orig,op,(800,800))
        # print(target)

    pts1 = np.float32([approx[0], approx[3], approx[1], approx[2]])
    pts2 = np.float32([[0, 0], [800, 0], [0, 800], [800, 800]])
    # print(approx[1])
    op = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img2, op, (800, 800))
    # cv2.imshow("Final",dst)
    # cv2.waitKey(0)
    cv2.imwrite("Final.jpg", dst)
    y = 287
    x = 148
    w = 489
    h = 73
    crop = dst[y:y + h, x:x + w]
    # cv2.imshow("cropped", crop)
    # cv2.waitKey(0)
    x = 152
    y = 424
    w = 489
    h = 66
    Depart = dst[y:y + h, x:x + w]
    # cv2.imshow("Department" , Depart )
    # cv2.waitKey(0)
    x = 276
    y = 713
    w = 117
    h = 32
    Enrol = dst[y:y + h, x:x + w]

    # cv2.imshow("Enrollment",Enrol)
    # cv2.waitKey(0)

    text = pytesseract.image_to_string(crop)
    text1 = pytesseract.image_to_string(Depart)

    if text !='' :
        detection=True

    print("NAME: ", text)
    print(text1)

    ## accessing google shee
    ## accessing google sheet ##
    phone,roll_no,year ,domain= '','','',''

    student_name = text

    data = sheet.get_all_records()
    for i in data:
        if i['name'] == student_name:
            phone = i['phone']
            roll_no = i['roll_no']
            year = i['year']
            domain = i['domain']

    for i in [phone,roll_no,year,domain]:
        if i=='':
            i = 'No data found'

    dictionary = {"student_name": student_name,'phone':phone,'roll_no':roll_no,'year':year,'domain':domain,'detection':detection}

    return render(request,"data_view.html",dictionary)





