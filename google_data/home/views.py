from django.shortcuts import render
import gspread
import oauth2client
from oauth2client.service_account import ServiceAccountCredentials

# scope = ['https://spreadsheets.google.com/feeds']
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']

creds = ServiceAccountCredentials.from_json_keyfile_name('IEEEWIE-9ea42a36af92.json', scope)
client = gspread.authorize(creds)

sheet = client.open('sample').sheet1

data = sheet.get_all_records()
# print(data)


# Create your views here.

def home(request):
    return render(request,"home.html")

def data_view(request):

        student_name = request.GET.get('student_name', '')

        for i in data:
            if i['name'] == student_name:
                phone = i['phone']

        dictionary = {"student_name": student_name,'phone':phone}
        return render(request, "data_view.html", dictionary)





