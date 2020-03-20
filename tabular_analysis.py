import pandas as pd
import csv
df = pd.read_csv('stream_out.csv')




data ={}

with open('out.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        elif row[2][0:2] == 'RT':
            print (row)
            data[row[4]] = []


with open('out.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        elif row[2][0:2] == 'RT':
            mySubString = row[2][row[2].find("@") + 1:row[2].find(":")]
            data[row[4]].append(mySubString)

print(data)