from django.shortcuts import render
import gspread
import oauth2client
from oauth2client.service_account import ServiceAccountCredentials
import cv2
import numpy as np
import pytesseract


# accessing google sheets
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('IEEEWIE-9ea42a36af92.json', scope)
client = gspread.authorize(creds)
sheet = client.open('sample').sheet1

# Create your views here.
def base(request):
    return render(request,"base.html")

def crop(dst,x,y,w,h):
    return dst[y:y + h, x:x + w]


def data_view(request):

    ## if student name given in text box----------------
    # ------------------------------------------------------------------
    student_name = request.GET.get('student_name', '')
    detection = False

    if student_name!='':
        phone, roll_no, year, domain = '', '', '', ''
        data = sheet.get_all_records()
        for i in data:
            if i['name'] == student_name:
                detection = True
                phone = i['phone']
                roll_no = i['roll_no']
                year = i['year']
                domain = i['domain']

        for i in [phone, roll_no, year, domain]:
            if i == '':
                i = 'No data found'

        dictionary = {"student_name": student_name, 'phone': phone, 'roll_no': roll_no, 'year': year, 'domain': domain,
                      'detection': detection}

        return render(request, "data_view.html", dictionary)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    # do detection
    MIN = 10
    img1 = cv2.imread('m.jpg', 0)
    detection = False
    video = cv2.VideoCapture(0)
    pic_taken = False
    name = ''

    while True:
        _, screen = video.read()
        cv2.imshow('pic', screen)

        if pic_taken:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                video.release()
                cv2.destroyAllWindows()
                break

        if not pic_taken:
            if cv2.waitKey(1) & 0xFF == ord('t'):
                pic_taken = True
                # _, img2 = video.read()
                img2 = screen
                print("pic captured")
                # img2 = cv2.imread('al.jpeg', 0)
                pic_taken = True
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
                    h, w = img1.shape
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    print("*********CARD MATCHED*********")
                else:
                    print("Not enough matches are found - %d/%d", len(good), MIN)
                try:
                    approx = np.int32(dst)
                except:
                    pass
    try:
        pts1 = np.float32([approx[0], approx[3], approx[1], approx[2]])
        pts2 = np.float32([[0, 0], [800, 0], [0, 800], [800, 800]])
        op = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img2, op, (800, 800))
        cv2.imwrite("Final.jpg", dst)

        name_crop = crop(dst,160,315,489,73)
        depart_crop = crop(dst,160,415,460,33)
        year_crop = crop(dst,160,450,520,35)
        roll_no_crop = crop(dst,245,501,310,34)

        name = pytesseract.image_to_string(name_crop)
        depart = pytesseract.image_to_string(depart_crop)
        year = pytesseract.image_to_string(year_crop)
        roll_no = pytesseract.image_to_string(roll_no_crop)

        if name != '':
            detection = True

        print("name: ", name)
        print("depart: ", depart)
        print("year : ", year)
        print("roll no : ", roll_no)

        try:
            b = roll_no.index('B') - 3
            roll = roll_no[0:b]
            batch_code = roll_no[-2::1]
            h = roll.index("-") + 1
            complete_rollno = roll[0: h] + batch_code + roll[h:]

        except:
            complete_rollno =''
            pass


    except:
        pass

    ## accessing google sheet ##
    print("complete roll no : ",complete_rollno)
    phone,roll_no,year ,domain= '','','',''
    data = sheet.get_all_records()
    for i in data:
        # if i['name'] == name:
        if i['roll_no'] == complete_rollno:
            name = i['name']
            phone = i['phone']
            roll_no = i['roll_no']
            year = i['year']
            domain = i['domain']

    dictionary = {"student_name": name,'phone':phone,'roll_no':roll_no,'year':year,'domain':domain,'detection':detection}
    return render(request,"data_view.html",dictionary)





