import cv2
import numpy as np
import pytesseract
import gspread
import oauth2client
from oauth2client.service_account import ServiceAccountCredentials
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']

creds = ServiceAccountCredentials.from_json_keyfile_name('IEEEWIE-9ea42a36af92.json', scope)
client = gspread.authorize(creds)

sheet = client.open('sample').sheet1



detection = False
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,', ' ,y)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # strXY = str(x) + ', '+ str(y)
        # cv2.putText(img, strXY, (x, y), font, .5, (255, 255, 0), 2)
        # cv2.imshow('image', img)



MIN = 10
img1 = cv2.imread('m.jpg', 0)
name, depart = '' ,''
detection = False
video = cv2.VideoCapture(0)
print("repeating")
while True:
    # _, img2 = video.read()
    img2 = cv2.imread('maryum2.jpeg', 0)
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

    cv2.imshow('pic', img2)
    cv2.setMouseCallback('pic', click_event)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        video.release()
        cv2.destroyAllWindows()
        break

    if len(good) > MIN:

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # img2 = cv2.polylines(img2, [np.int32(dst)], True, (255), 3, cv2.LINE_AA)
        print("*********CARD MATCHED*********")
    else:
        print("Not enough matches are found - %d/%d" %
              (len(good), MIN))
        # matchesMask = None
    # draw_params = dict(matchColor=(0, 255, 0),
    #                    singlePointColor=None,
    #                    matchesMask=matchesMask,
    #                    flags=2)
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    # plt.imshow(img3, 'gray'), plt.show()
    # print(dst)


try:
    approx = np.int32(dst)
except:
    pass


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
cv2.imshow("Final",dst)
cv2.setMouseCallback('Final', click_event)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
cv2.imwrite("Final.jpg", dst)

y = 315 #287  #25
# y = 287 +25
x = 160 #10
# x = 148 +10
w = 489
h = 73
name_crop = dst[y:y + h, x:x + w]
cv2.imshow("name", name_crop)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
# x = 152
# y = 424
# w = 489
# h = 66
x= 160
y =415
h = 33
w = 460
depart_crop = dst[y:y + h, x:x + w]
cv2.imshow("Department" , depart_crop )
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()

# x = 276 ##70
# y = 713 #-10
# w = 117
# h = 32
x = 160
y = 450
h = 35
w = 520

year_crop = dst[y:y + h, x:x + w]

cv2.imshow("year",year_crop)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()


x = 160
y=500
w = 310
h = 33
roll_no_crop = dst[y:y + h, x:x + w]

cv2.imshow("roll_no", roll_no_crop)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()

name = pytesseract.image_to_string(name_crop)
depart = pytesseract.image_to_string(depart_crop)
year = pytesseract.image_to_string(year_crop)
roll_no = pytesseract.image_to_string(roll_no_crop)


if name !='' :
    detection=True

print("name: ", name)
print("depart: ", depart)
print("year : ", year)
print("roll no : ", roll_no)

# print(len(name))
# student_name = name
#
# data = sheet.get_all_records()
# for i in data:
#     if i['name'] == student_name:
#         phone = i['phone']
#         roll_no = i['roll_no']
#         year = i['year']
#         domain = i['domain']
#
# print(" data  from sheet /n ------" , student_name,phone,roll_no,year,domain ,"------")






