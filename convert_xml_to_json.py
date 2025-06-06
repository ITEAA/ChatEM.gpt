import xmltodict
import json

with open("job_data.xml", "r", encoding="utf-8") as f:
    xml_str = f.read()

data_dict = xmltodict.parse(xml_str)
json_data = json.dumps(data_dict, indent=2, ensure_ascii=False)

with open("job_data.json", "w", encoding="utf-8") as f:
    f.write(json_data)

print("✅ 변환 완료: job_data.json")
