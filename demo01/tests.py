from django.test import TestCase

# Create your tests here.
import datetime
# print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

in_date = '2022-10-31 00:00:00'
dt = datetime.datetime.strptime(in_date, "%Y-%m-%d %H:%M:%S")
out_date = (dt + datetime.timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
print(out_date)
