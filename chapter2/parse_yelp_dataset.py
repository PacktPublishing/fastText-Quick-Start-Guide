import csv
import sys
w = csv.writer(sys.stdout)
for row in csv.DictReader(sys.stdin):
    w.writerow([row['stars'], row['text'].replace('\n', '')])
