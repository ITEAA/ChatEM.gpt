import xml.etree.ElementTree as ET
import json

with open("response.xml", encoding="utf-8") as f:
    xml_data = f.read()

root = ET.fromstring(xml_data)
jobs = []
for item in root.findall(".//jobList"):
    jobs.append({
        "name": item.findtext("entrprsNm"),
        "title": item.findtext("pblancSj"),
        "area": item.findtext("areaStr"),
    })

print(json.dumps(jobs, indent=2, ensure_ascii=False))
