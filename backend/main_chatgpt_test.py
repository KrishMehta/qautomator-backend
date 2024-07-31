import requests
import json
import time
import re

base_url = "http://127.0.0.1:8000"

video_file_path = "/Users/krish/Downloads/sample_video_pnr_status.mp4"


def extract_test_cases(code_snippet):
    """
    Extract test case functions from the provided code snippet.
    
    :param code_snippet: The entire code snippet as a string.
    :return: A string containing all test case functions.
    """
    # Regular expression to match class methods starting with 'test_case_' and include all indented lines
    test_case_pattern = re.compile(r'(def test_case_\d+:\n(?: {4}.*\n)*)', re.MULTILINE)

    # Find all matches in the provided code snippet
    matches = test_case_pattern.findall(code_snippet)

    # Join all matches with a newline to form the output string
    result = "\n".join(matches)

    return result


start_time = time.time()
with open(video_file_path, "rb") as file:
    response = requests.post(f"{base_url}/func_flow/", files={"file": file})
    response.raise_for_status()
    func_flow_result = response.json()["result"]
end_time = time.time()
step_1_duration = end_time - start_time
print("Step 1 Result:", func_flow_result)
print(f"Step 1 Duration: {step_1_duration:.2f} seconds")
print("\n")

start_time = time.time()
with open(video_file_path, "rb") as file:
    response = requests.post(
        f"{base_url}/generate_test_cases/",
        files={"file": file},
        data={"application_flow": func_flow_result}
    )
    response.raise_for_status()
    test_cases_result = response.json()["result"]
end_time = time.time()
step_2_duration = end_time - start_time
print("Step 2 Result:", test_cases_result)
print(f"Step 2 Duration: {step_2_duration:.2f} seconds")
print("\n")

start_time = time.time()
with open(video_file_path, "rb") as file:
    response = requests.post(
        f"{base_url}/generate_test_cases_code",
        files={"file": file},
        data={
            "application_flow": func_flow_result,
            "test_cases_list": json.dumps(test_cases_result),
            "os_type": "android"
        }
    )
    response.raise_for_status()
    test_cases_code_result = response.json()["result"]
    extracted_test_cases = extract_test_cases(test_cases_code_result)
end_time = time.time()
step_3_duration = end_time - start_time
print("Step 3 Result:", test_cases_code_result)
print(f"Step 3 Duration: {step_3_duration:.2f} seconds")
print("\n")
