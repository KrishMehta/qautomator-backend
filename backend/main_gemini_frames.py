from typing import List

from fastapi import FastAPI, File, UploadFile, Form
import cv2
import base64
import tempfile
import os
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import multiprocessing

from qautomate.helpers.visual_testing_helper import process_color_in_images, process_layout_in_images, visual_analyze, \
    process_text_in_images

import google.generativeai as genai
import time

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

gemini_key = os.getenv('gemini_key')
if not gemini_key:
    raise ValueError("No Gemini API key found in environment variable 'GOOGLE_API_KEY'")

genai.configure(api_key=gemini_key)

#
# class TestCaseCodeGenRequest(BaseModel):
#     test_no: str
#     test_details: str


all_screens = {
    "home_screen": "The main screen allows users to book trains, flights, buses, and hotels, with search fields for train routes, date selection, and options for checking running status and PNR status. Key UI elements include tabs for different transportation modes, search functionality, and quick access to services like seat availability and food orders.",
    "pnr_status_screen": "This screen allows users to check their train PNR status by entering their 10-digit PNR number. It also provides quick access to features like coach and seat information, platform locator, refund calculator, and ixigo AU card services.",
    "search_result_page_trains": "This screen displays available train options for a selected route and date, showing train names, departure and arrival times, travel duration, and fare details. It includes filters for best available and AC-only options, along with seat availability and schedule links for each train."
}

# Load mapper.json
with open('qautomate/screen_ui_elements_map.json', 'r') as f:
    screen_mapper_android = json.load(f)
    # print(screen_mapper)

with open('qautomate/screen_ui_elements_map_ios.json', 'r') as f:
    screen_mapper_ios = json.load(f)

with open('qautomate/screens_for_visual_testing.json', 'r') as f:
    visual_testing_images = json.load(f)


async def capture_frames_at_intervals(video_file, interval_ms=1000):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await video_file.read())
            tmp_path = tmp.name

        video = cv2.VideoCapture(tmp_path)

        if not video.isOpened():
            raise IOError(f"Error: Could not open video file {tmp_path}")

        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)  # Frame rate
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms = int((total_frames / fps) * 1000)  # Total duration in milliseconds

        print(f"Video FPS: {fps}")
        print(f"Total frames: {total_frames}")
        print(f"Estimated duration (milliseconds): {duration_ms}")

        base64Frames = []
        frame_count = 0

        for ms in range(0, duration_ms + 1, interval_ms):  # Include the last frame
            video.set(cv2.CAP_PROP_POS_MSEC, ms)  # Set the position of the video in milliseconds
            success, frame = video.read()
            if success:
                _, buffer = cv2.imencode(".jpg", frame)
                base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
                frame_count += 1
            else:
                print(f"Warning: Frame at {ms} ms could not be read.")

        print(f"{frame_count} frames read and encoded (every {interval_ms} ms).")
        return base64Frames

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        video.release()


@app.post("/func_flow_gemini/")
async def generate_func_flow_gemini(file: UploadFile = File(...)):
    print("inside generate_func_flow")
    base64Frames = await capture_frames_at_intervals(file, 1000);
    print("no of frames", len(base64Frames))
    # Create the prompt
    prompt = '''I am testing an Android application using video analysis to understand its functionality of 
                scheduled flight status flow. Specifically, I need to analyze the video frames of the application to generate a detailed functionality flow based on user interactions, focusing on the static UI elements and predefined states.

                    Please follow these steps:

                    1. **Analyze the Video Frames:**
                       - Observe the video frames to identify the sequence of user interactions with the application.
                       - Note down each step in detail, including any screen transitions, user inputs, and system responses, focusing on static UI elements and not on dynamic data from API responses.

                    2. **Generate the Functional Flow:**
                       - Provide a detailed flow of the feature based on the observed interactions in the video frames.
                       - Clearly depict each step, including relevant conditions or branching logic triggered by user interactions or predictable system responses.
                       - Ensure that each step is described in the order it occurs, emphasizing static elements like buttons, input fields, labels, and predefined messages.

                    Example Structure:

                    **Functional Flow:**
                    - Step 1: [Description of user interaction and initial state, e.g., "User launches the app and observes the splash screen."]
                    - Step 2: [Description of subsequent interaction and app response, focusing on static elements, e.g., "User navigates to the main screen and sees options for Trains, Flights, Buses, and Hotels."]
                    - Step 3: [Repeat for each step observed, describing user interactions and static components only.]

                    Focus: Capture the functionality flow based on user interactions as seen in the video frames of the application, 
                    concentrating on static UI elements and avoiding reliance on dynamic data. Each step should be clear and concise, 
                    capturing the essence of user actions and predictable app behavior.
                    
    These are frames from a video that I want to upload.'''

    # Set the model to Gemini 1.5 Pro.
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

    # Make the LLM request.
    print("Making LLM inference request...")
    response = model.generate_content([prompt, base64Frames],
                                      request_options={"timeout": 600})

    total_tokens = response.usage_metadata.total_token_count
    print("Total tokens used", total_tokens)

    print("result", response.text)
    return {"result": response.text}


@app.post("/generate_test_cases_gemini/")
async def generate_test_cases_gemini(file: UploadFile = File(...),
                                     application_flow: str = Form(...),
                                     type_of_flow: str = Form(...)):
    print("generating TCs")
    base64Frames = await capture_frames_at_intervals(file, 1000)
    # Create the prompt
    prompt = f'''Based on the detailed functionality flow generated from the video frames,
                I need to create comprehensive UI-based test cases for the functionality of {type_of_flow}, based on the detailed functionality flow generated from the video frames.

                Please make sure that the test cases are comprehensive, covering all possible scenarios as observed in the video frames, and have expected outcomes based on static UI elements in the video frames and not dependent on dynamic data from APIs.
                All Screens available {all_screens}
                
                ### Instructions:

                1. **Review the Functional Flow:**
                   - Carefully review the functionality flow generated from the video analysis.
                   - Understand each interaction and the corresponding system response.
                   - Focus on the static UI elements involved in each interaction (e.g., buttons, input fields, labels) rather than dynamic content.
                    {application_flow}

                2. **Generate Test Cases:**
                   - Create detailed UI test cases for each interaction with the application.
                   - Ensure each test case includes:
                     - A clear and specific description of the test case
                     - Attach the impacted screen names along with it.
                     - Step-by-step instructions based on observed interactions in the video frames
                     - Specific and measurable expected outcomes based solely on static UI elements (e.g., presence of buttons, input fields, labels)
                     - Identification of edge cases and potential user errors (e.g., incorrect inputs, system errors)


                    Example Structure:

                    **Test Case 1:**
                    - **Description:** [Detailed description of the test case]
                    - **Impacted Screens:** [screen1, screen2]
                    - **Steps:**
                      1. [Step-by-step instructions]
                      2. [Continue steps as necessary]
                    - **Expected Outcome:** [Specific expected results that can be validated using UI elements]

                    **Test Case 2:**
                    - **Description:** [Another detailed description]
                    - **Impacted Screens:** [screen1]
                    - **Steps:**
                      1. [Step-by-step instructions]
                      2. [Continue steps]
                    - **Expected Outcome:** [Specific expected results]

    These are frames from a video that I want to upload.'''

    # Set the model to Gemini 1.5 Pro.
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

    # Make the LLM request.
    print("Making LLM inference request...")
    response = model.generate_content([prompt, base64Frames],
                                      request_options={"timeout": 600})
    print("result", response.text)

    total_tokens = response.usage_metadata.total_token_count
    print("Total tokens used", total_tokens)

    return {"result": response.text}


@app.post("/generate_test_cases_code_gemini")
async def generate_code_for_test_cases_gemini(file: UploadFile = File(...),
                                              application_flow: str = Form(...),
                                              type_of_flow: str = Form(...),
                                              test_cases_list: str = Form(...),
                                              os_type: str = Form(...)):
    print("TC list", test_cases_list)
    print("OS", os_type)

    print(type(test_cases_list))

    test_case_list_obj = json.loads(test_cases_list)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        video_path = tmp.name

    # Upload the video file
    print(f"Uploading file...")
    video_file = genai.upload_file(path=video_path, mime_type="video/mov")
    print(f"Completed upload: {video_file.uri}")

    # Wait for the video file to be processed
    while video_file.state.name == "PROCESSING":
        print('.', end='')
        time.sleep(10)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)

    impacted_screens = set()
    for test_case in test_case_list_obj:
        # print(test_case)
        lines = test_case.split("\n")
        # print(lines)
        for line in lines:
            if line.startswith("- **Impacted Screens:**"):
                screens = line.replace("- **Impacted Screens:**", "").strip().split(", ")
                impacted_screens.update(screens)

    # Fetch the screen data from mapper.json
    screen_mapper = screen_mapper_android if os_type == "android" else screen_mapper_ios
    screen_data = {screen: screen_mapper.get(screen, {}) for screen in impacted_screens}

    # print("Impacted Screens:", impacted_screens)
    print("Screen Data:", screen_data)

    # Create the prompt
    prompt_android = f'''I have developed test cases for an Android application based on the {type_of_flow} functionality.
                Using the provided video, functional flow, the xpaths of UI elements, and test cases,
                I need to generate Appium code with appropriate assertions and comments and automate the testing of this
                specific functionality.

                Note that widgets in the app can be EditText, TextView, Button or some other type, so use XPath given in the prompt 
                to locate these components. Make sure to use static UI elements not dependent on dynamic data from APIs.

                Here are the key details:

                ### Functional Flow: 
                        {application_flow}

                ### Impacted Screens UI Element XPATHS to prepare testcases :
                        {screen_data}
                        
                ### Test Cases:
                        {test_cases_list}

                    Example Appium Code:

                    def test_case_1(self):
                        """
                        Verify the "PNR Status" option is visible and selectable on the Home screen.
                        Steps:
                            1. Navigate to the Home screen.
                            2. Locate the "PNR Status" button.
                            3. Tap on the "PNR Status" button.
                        Expected Outcome: User is navigated to the PNR Status screen.
                        """
                        # Wait for the Home screen to load and locate the "PNR Status" button
                        WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.XPATH, "//android.widget.TextView[@text='PNR Status']"))
                        )
                        pnr_status_button = self.driver.find_element(By.XPATH, "//android.widget.TextView[@text='PNR Status']")
                        self.assertTrue(pnr_status_button.is_displayed())

                        # Tap on the "PNR Status" button
                        pnr_status_button.click()

                        # Wait for the PNR Status screen to load
                        WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.XPATH, "//android.widget.EditText[@text='Enter your 10 digit PNR']"))
                        )
                        self.assertTrue(self.driver.find_element(By.XPATH,
                                                                 "//android.widget.EditText[@text='Enter your 10 digit PNR']").is_displayed())

                    def test_case_2(self):
                        """
                        Verify user can enter a valid PNR number.
                        Steps:
                            1. Navigate to the PNR Status screen.
                            2. Enter a valid 10-digit PNR number in the input field.
                            3. Tap the "Search" button.
                        Expected Outcome: The input field accepts the PNR number, and the user can tap the search button.
                        """
                        # Navigate to the PNR Status screen (same steps as Test Case 1)
                        self.test_case_1()

                        # Locate the PNR input field and enter a valid 10-digit PNR number
                        pnr_input_field = self.driver.find_element(By.XPATH,
                                                                   "//android.widget.EditText[@text='Enter your 10 digit PNR']")
                        pnr_input_field.send_keys("1234567890")

                        # Locate the "Search" button and tap on it
                        search_button = self.driver.find_element(By.XPATH, "//android.widget.Button[contains(@text, 'Search')]")
                        search_button.click()

                    def test_case_3(self):
                        """
                        Verify the application handles invalid PNR numbers.
                        Steps:
                            1. Enter an invalid or less than 10-digit PNR number.
                        Expected Outcome: The search button should remain disabled.
                        """
                        # Navigate to the PNR Status screen (same steps as Test Case 1)
                        self.test_case_1()

                        # Locate the PNR input field and enter an invalid PNR number (less than 10 digits)
                        pnr_input_field = self.driver.find_element(By.XPATH,
                                                                   "//android.widget.EditText[@text='Enter your 10 digit PNR']")
                        pnr_input_field.send_keys("12345")

                        # Locate the "Search" button and tap on it
                        search_button = self.driver.find_element(By.XPATH, "//android.widget.Button[contains(@text, 'Search')]")
                        # search_button.click()
                        # Verify that the button is disabled
                        assert not search_button.is_enabled(), "The search button should be disabled for invalid PNR input"

    This is the video that I want to upload.'''

    # Create the prompt
    prompt_ios = f'''I have developed test cases for an iOS application based on the {type_of_flow} functionality.
                Using the provided video, functional flow, the xpaths of UI elements, and test cases,
                I need to generate Appium code in JavaScript with appropriate assertions and comments and automate the testing of this
                specific functionality.

                Note that widgets in the app can be EditText, TextView, Button or some other type, so use XPath given in the prompt 
                to locate these components. Make sure to use static UI elements not dependent on dynamic data from APIs.

                Here are the key details:

                ### Functional Flow: 
                        {application_flow}

                ### Impacted Screens UI Element XPATHS to prepare testcases :
                        {screen_data}
                        
                ### Test Cases:
                        {test_cases_list}

                    Example Appium Code for iOS:

                    async function testCase1(client) {{
                        try {{
                            // Verify the "PNR Status" option is visible and selectable on the Home screen.
                            // Steps:
                            // 1. Navigate to the Home screen.
                            // 2. Locate the "PNR Status" button.
                            // 3. Tap on the "PNR Status" button.
                            // Expected Outcome: User is navigated to the PNR Status screen.

                            // Wait for the Home screen to load and locate the "PNR Status" button
                            const pnrStatusButton = await client.$('//XCUIElementTypeStaticText[contains(@label, "PNR Status")]');
                            if (await pnrStatusButton.isDisplayed()) {{
                                await pnrStatusButton.click();
                            }} else {{
                                throw new Error('"PNR Status" button is not displayed');
                            }}

                            // Wait for the PNR Status screen to load
                            await client.pause(2000); // Adjust the delay as necessary

                            // Validate the PNR Status screen
                            const pnrInput = await client.$('//XCUIElementTypeTextField[contains(@value, "Enter your 10 digit PNR")]');
                            if (!await pnrInput.isDisplayed()) {{
                                throw new Error('"Enter your 10 digit PNR" input field is not displayed');
                            }}
                        }} catch (error) {{
                            console.error('Error in testCase1:', error);
                            throw error;
                        }}
                    }}

                    async function testCase2(client) {{
                        try {{
                            // Verify user can enter a valid PNR number.
                            // Steps:
                            // 1. Navigate to the PNR Status screen.
                            // 2. Enter a valid 10-digit PNR number in the input field.
                            // 3. Tap the "Search" button.
                            // Expected Outcome: The input field accepts the PNR number, and the user can tap the search button.

                            // Navigate to the PNR Status screen (reuse steps from testCase1)
                            await testCase1(client);

                            // Locate the PNR input field and enter a valid 10-digit PNR number
                            const pnrInput = await client.$('//XCUIElementTypeTextField[contains(@value, "Enter your 10 digit PNR")]');
                            await pnrInput.setValue('1234567890');

                            // Locate and tap the "Search" button
                            const searchButton = await client.$('//XCUIElementTypeButton[contains(@label, "Search")]');
                            await searchButton.click();

                            // Wait for the search results to load
                            await client.pause(5000); // Adjust the delay as necessary
                        }} catch (error) {{
                            console.error('Error in testCase2:', error);
                            throw error;
                        }}
                    }}

                    async function testCase3(client) {{
                        try {{
                            // Verify the application handles invalid PNR numbers.
                            // Steps:
                            // 1. Enter an invalid or less than 10-digit PNR number.
                            // Expected Outcome: The search button should remain disabled.

                            // Navigate to the PNR Status screen (reuse steps from testCase1)
                            await testCase1(client);

                            // Locate the PNR input field and enter an invalid PNR number (less than 10 digits)
                            const pnrInput = await client.$('//XCUIElementTypeTextField[contains(@value, "Enter your 10 digit PNR")]');
                            await pnrInput.setValue('12345');

                            // Locate the "Search" button
                            const searchButton = await client.$('//XCUIElementTypeButton[contains(@label, "Search")]');

                            // Verify that the search button is disabled
                            const isEnabled = await searchButton.isEnabled();
                            if (isEnabled) {{
                                throw new Error('The search button should be disabled for invalid PNR input');
                            }}
                        }} catch (error) {{
                            console.error('Error in testCase3:', error);
                            throw error;
                        }}
                    }}

    This is the video that I want to upload.'''

    # print(PROMPT_MESSAGES)

    # Set the model to Gemini 1.5 Pro.
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

    # Make the LLM request.
    print("Making LLM inference request...")
    if os_type == 'android':
        response = model.generate_content([prompt_android, video_file],
                                          request_options={"timeout": 600})
    else:
        response = model.generate_content([prompt_ios, video_file],
                                          request_options={"timeout": 600})
    print("result", response.text)
    return {"result": response.text}


class VisualTestingRequest(BaseModel):
    testScreen: str
    screen_type: str
    osType: str


class BackendTestingRequest(BaseModel):
    curl: str
    functionality: str
    success_response: str
    error_response: str
    test_cases_list: list


def worker(args):
    func, original_image, test_screen = args
    return func(original_image, test_screen)


@app.post("/visual_testing_gemini")
async def visual_testing_gemini(request: VisualTestingRequest):
    # print(request.testScreen)
    # print(request.screen_type)
    # print(request.osType)
    print("visual testing stared")
    original_image = visual_testing_images.get(request.osType).get(request.screen_type)
    # Define the functions and their arguments
    tasks = [
        (process_color_in_images, original_image, request.testScreen),
        (process_layout_in_images, original_image, request.testScreen),
        (process_text_in_images, original_image, request.testScreen)
    ]

    # Run the functions in parallel using multiprocessing
    with multiprocessing.Pool(processes=3) as pool:
        results = pool.map(worker, tasks)

    # Collect the results
    color_diff, layout_diff, text_diff = results

    visual_testing_result = visual_analyze(text_diff, layout_diff, color_diff)
    # print(visual_testing_result)
    return {"result": {
        "visual_testing": visual_testing_result,
        "original_Img": original_image,
        "testing_Img": request.testScreen
    }}
    # return {"result": {
    #     "color": color_diff,
    #     "layout": layout_diff,
    #     "text": text_diff
    # }}


@app.post("/backend_tc_gen_gemini")
async def generate_test_cases_for_backend_gemini(request: BackendTestingRequest):
    print("curl", request.curl)

    # Create the prompt
    prompt = f'''Based on the functionality : {request.functionality}, the cURL request: {request.curl},
                the success : {request.success_response} and the error response : {request.error_response}
                I need to create comprehensive backend test cases so that each scenario can be validated.

                Please make sure that the test cases are comprehensive, covering all possible scenarios.
                
                ### Instructions:

                1. **Review the Functional Flow:**
                   - Carefully understand and review the functionality flow given.
                    {request.functionality}

                2. **Generate Test Cases:**
                   - Create detailed backend test cases for each scenario.
                   - Follow the format given strictly in example only.
                   - Ensure each test case includes:
                     - A clear and specific description of the test case
                     - Any setup or preconditions that must be met before the test can be executed (e.g., database state, specific configurations).
                     - Detailed steps to execute the test, including HTTP method (GET, POST, PUT, DELETE), endpoint URL, headers, and request body.
                     - Specific input data required for the test case, such as query parameters, path variables, or JSON payloads.
                     - The expected outcome of the test, including status codes, response body structure, response headers, and any other relevant details.

                    Example Structure + Format:

                    ###Test Case 1:###
                    - **Name:** [Name of the testcase]
                    - **Description:** [Detailed description of the test case]
                    - **Preconditions:** [Any setup or preconditions that must be met before the test can be executed ]
                    - **Test Steps:**
                      1. [Step-by-step instructions]
                      2. [Continue steps as necessary]
                    - **Test Data** [Specific input data required for the test case, such as query parameters, path variables, or JSON payloads.]
                    - **Expected Result:** [The expected outcome of the test, including status codes, response body structure, response headers, and any other relevant details.]

                    ###Test Case 2:###
                    - **Name:** [Name of the testcase]
                    - **Description:** [Detailed description of the test case]
                    - **Preconditions:** [Any setup or preconditions that must be met before the test can be executed ]
                    - **Test Steps:**
                      1. [Step-by-step instructions]
                      2. [Continue steps as necessary]
                    - **Test Data** [Specific input data required for the test case, such as query parameters, path variables, or JSON payloads.]
                    - **Expected Result:** [The expected outcome of the test, including status codes, response body structure, response headers, and any other relevant details.]
                '''

    # Set the model to Gemini 1.5 Pro.
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

    # Make the LLM request.
    print("Making LLM inference request...")
    response = model.generate_content(prompt,
                                      request_options={"timeout": 600})
    print("result", response.text)
    return {"result": response.text}


@app.post("/backend_tc_code_gen_gemini")
async def generate_test_cases_code_for_backend_gemini(request: BackendTestingRequest):
    # Create the prompt
    prompt = f'''I have developed test cases for by backend application based on the functionality : {request.functionality}, the cURL request: {request.curl},
                the success : {request.success_response} and the error response : {request.error_response}
                
                I need to generate java code for these test cases with appropriate assertions and comments and automate the testing of this
                specific functionality.

                Please make sure that the test cases are comprehensive, covering all possible scenarios.
                
                ### Test Cases:
                {request.test_cases_list}
                
                ### Example Test Cases codes for reference:
                
                ''' + '''
    /**
     * Test Case: Valid PNR Number
     * Description: Verify that the API returns correct details when a valid PNR number is provided.
     * Preconditions: A valid PNR number exists in the system.
     */
    @Test
    public void testValidPnrNumber() {
        String validPnr = "1234567890";
        String mode = "NEW_ADDITION";

        // Send a GET request to the endpoint with valid PNR number
        Response response = given()
                .queryParam("pnr", validPnr)
                .queryParam("mode", mode)
                .when()
                .get("/pnr/enquiry")
                .then()
                .statusCode(200)
                .extract()
                .response();

        // Validate response headers
        assertEquals("application/json", response.getHeader("Content-Type"));

        // Validate response body
        response.then().body("data.id", equalTo(validPnr));
        response.then().body("data.name", equalTo("TRAIN " + validPnr));
        // Add more assertions as needed to validate the response body
    }

    /**
     * Test Case: Invalid PNR Number
     * Description: Verify that the API returns an error message when an invalid PNR number is provided.
     * Preconditions: The PNR number does not exist in the system.
     */
    @Test
    public void testInvalidPnrNumber() {
        String invalidPnr = "0000000000";
        String mode = "NEW_ADDITION";

        // Send a GET request to the endpoint with invalid PNR number
        Response response = given()
                .queryParam("pnr", invalidPnr)
                .queryParam("mode", mode)
                .when()
                .get("/pnr/enquiry")
                .then()
                .statusCode(500)
                .extract()
                .response();

        // Validate response headers
        assertEquals("application/json", response.getHeader("Content-Type"));

        // Validate response body
        response.then().body("errors.code", equalTo(500));
        response.then().body("errors.message", equalTo("PNR No. is not valid"));
    }'''

    # Set the model to Gemini 1.5 Pro.
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

    # Make the LLM request.
    print("Making LLM inference request...")
    response = model.generate_content(prompt,
                                      request_options={"timeout": 600})
    print("result", response.text)
    return {"result": response.text}
