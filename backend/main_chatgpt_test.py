import requests
import json
import time

base_url = "http://127.0.0.1:8000"

video_file_path = "/Users/krish/Downloads/QAutomator.mov"

start_time = time.time()
with open(video_file_path, "rb") as file:
    response = requests.post(f"{base_url}/func_flow/", files={"file": file})
    response.raise_for_status()
    func_flow_result = response.json()["result"]
end_time = time.time()
step_1_duration = end_time - start_time
print("Step 1 Result:", func_flow_result)
print(f"Step 1 Duration: {step_1_duration:.2f} seconds")

start_time = time.time()
with open(video_file_path, "rb") as file:
    response = requests.post(
        f"{base_url}/generate_test_cases/",
        files={"file": file},
        data={"application_flow": func_flow_result, "type_of_flow": "scheduled flight status flow"}
    )
    response.raise_for_status()
    test_cases_result = response.json()["result"]
end_time = time.time()
step_2_duration = end_time - start_time
print("Step 2 Result:", test_cases_result)
print(f"Step 2 Duration: {step_2_duration:.2f} seconds")

start_time = time.time()
with open(video_file_path, "rb") as file:
    response = requests.post(
        f"{base_url}/generate_test_cases_code",
        files={"file": file},
        data={
            "application_flow": func_flow_result,
            "type_of_flow": "scheduled flight status flow",
            "test_cases_list": json.dumps(test_cases_result),
            "os_type": "ios"
        }
    )
    response.raise_for_status()
    test_cases_code_result = response.json()["result"]
end_time = time.time()
step_3_duration = end_time - start_time
print("Step 3 Result:", test_cases_code_result)
print(f"Step 3 Duration: {step_3_duration:.2f} seconds")
