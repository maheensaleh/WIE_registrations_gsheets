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

def base(request):
    return render(request,"base.html")


def filter(request):
    domain = request.GET.get('domain', '')
    param = {'domain': domain}

    return render(request,"filter.html",param)

def data_view(request):
        phone = ''
        roll_no = ''
        year = ''
        domain= ''

        student_name = request.GET.get('student_name', '')

        for i in data:
            if i['name'] == student_name:
                phone = i['phone']
                roll_no = i['roll_no']
                year = i['year']
                domain = i['domain']

        for i in [phone,roll_no,year,domain]:
            if i=='':
                i = 'No data found'

        dictionary = {"student_name": student_name,'phone':phone,'roll_no':roll_no,'year':year,'domain':domain}
        return render(request, "data_view.html", dictionary)





