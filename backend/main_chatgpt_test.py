import json
import logging
import re
import requests
import time

from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.webdriver.common.appiumby import AppiumBy
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logging.basicConfig(level=logging.INFO)

base_url = "http://127.0.0.1:8000"

video_file_path = "/Users/krish/Downloads/sample_video_pnr_status.mp4"

driver = None


def setup():
    global driver
    options = UiAutomator2Options()
    options.platform_name = 'Android'
    options.device_name = 'emulator-5556'
    options.app_package = 'com.ixigo.train.ixitrain'
    options.app_activity = 'com.ixigo.train.ixitrain.TrainActivity'
    options.no_reset = True

    try:
        driver = webdriver.Remote('http://localhost:4723/wd/hub', options=options)

        # Navigate to the home screen
        driver.press_keycode(3)

        # Open the app drawer
        driver.swipe(start_x=500, start_y=1500, end_x=500, end_y=500, duration=800)
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((AppiumBy.XPATH, "//android.widget.TextView[@text='ixigo trains']"))
        )

        # Click on the ixigo app icon
        ixigo_icon = driver.find_element(AppiumBy.XPATH, "//android.widget.TextView[@text='ixigo trains']")
        ixigo_icon.click()

        # Wait until the app is launched
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((AppiumBy.XPATH, "//android.widget.TextView[@text='Trains']"))
        )

    except Exception as e:
        logging.error(f"Error setting up Appium driver: {e}")
        raise


def teardown():
    global driver
    if driver:
        driver.quit()


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
end_time = time.time()
step_3_duration = end_time - start_time
print("Step 3 Result:", test_cases_code_result)
print(f"Step 3 Duration: {step_3_duration:.2f} seconds")
print("\n")

extracted_test_cases = extract_test_cases(test_cases_code_result)
print(f"Extracted test cases: {extracted_test_cases}")
# try:
#     setup()
#     exec(extracted_test_cases)
# except Exception as e:
#     logging.error(f"Error during test execution: {e}")
# finally:
#     teardown()
