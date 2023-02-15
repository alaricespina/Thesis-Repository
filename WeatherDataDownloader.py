
import requests
import sys

import json
                
LOCATION = "Quezon City Philippines"
REQ_LOCATION = LOCATION.replace(" ", "%20")
API_KEY = 'TPM24WCAKMFSLZDCC5APU52JD'
DATE_START = "2022-01-01"
DATE_END =  "2023-01-01"

response = requests.request("GET", f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{REQ_LOCATION}/{DATE_START}/{DATE_END}?unitGroup=us&key={API_KEY}&contentType=json")
if response.status_code!=200:
  print('Unexpected Status code: ', response.status_code)
  sys.exit()  


# Parse the results as JSON
jsonData = response.json()
        
