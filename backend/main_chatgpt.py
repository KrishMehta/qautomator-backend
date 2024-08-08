import base64
import json
import logging
import os
import re
import shutil
import tempfile
import uuid

import cv2
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from skimage.metrics import structural_similarity as ssim

from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.webdriver.common.appiumby import AppiumBy
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Read the value of the environment variable
openai_key = os.getenv('openai_key')
if not openai_key:
    raise ValueError("No OpenAI API key found in environment variable 'openai_key'")
client = OpenAI(api_key=openai_key)

input_cost_per_million = 0.15  # 15 cents per 1M input tokens
output_cost_per_million = 0.60  # 60 cents per 1M output tokens

all_screens = {
    "home_screen": "This screen allows users to book trains, flights, buses, and hotels, with search fields for train routes, date selection, and options for checking running status and PNR status. Key UI elements include tabs for different transportation modes, search functionality, and quick access to services like seat availability and food orders.",
    "pnr_status_screen": "This screen allows users to check their train PNR status by entering their 10-digit PNR number. It also provides quick access to features like coach and seat information, platform locator, refund calculator, and ixigo AU card services.",
    "srp_screen": "This screen displays available train options for a selected route and date, showing train names, departure and arrival times, travel duration, and fare details. It includes filters for best available and AC-only options, along with seat availability and schedule links for each train."
}

with open('qautomate/screen_ui_elements_map.json', 'r') as f:
    screen_mapper_android = json.load(f)

with open('qautomate/screen_ui_elements_map_ios.json', 'r') as f:
    screen_mapper_ios = json.load(f)

base64_collage = None

database = {
    "tests": {},
    "func_flows": {},
    "test_cases": {},
    "test_cases_code": {}
}

driver = None


async def capture_frames_at_intervals(video_file, interval_ms=250, max_frames=25):
    global base64_collage
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        video = cv2.VideoCapture(tmp_path)

        if not video.isOpened():
            raise IOError(f"Error: Could not open video file {tmp_path}")

        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms = int((total_frames / fps) * 1000)

        logger.info(f"Video FPS: {fps}")
        logger.info(f"Total frames: {total_frames}")
        logger.info(f"Estimated duration (milliseconds): {duration_ms}")

        frames = []
        frame_count = 0
        previous_frame_gray = None

        # Create directories to save images
        all_frames_dir = os.path.abspath('./all_frames')
        if not os.path.exists(all_frames_dir):
            os.makedirs(all_frames_dir)

        output_frames_dir = os.path.abspath('./output_frames')
        if not os.path.exists(output_frames_dir):
            os.makedirs(output_frames_dir)

        # Capture the first frame
        video.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = video.read()
        if success:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
            frame_count += 1

            logger.info("Frame at 0 ms added")

            previous_frame_gray = frame_gray

            # Save the first frame
            frame_path = os.path.join(output_frames_dir, 'frame_0.jpg')
            plt.figure()
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title('Frame at 0 ms')
            plt.axis('off')
            plt.savefig(frame_path)
            plt.close()

            all_frames_path = os.path.join(all_frames_dir, 'frame_0.jpg')
            plt.figure()
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title('Frame at 0 ms')
            plt.axis('off')
            plt.savefig(all_frames_path)
            plt.close()
        else:
            logger.warning("Warning: First frame could not be read.")

        ssim_differences = []

        for ms in range(interval_ms, duration_ms + 1, interval_ms):  # Start from interval_ms
            video.set(cv2.CAP_PROP_POS_MSEC, ms)  # Set the position of the video in milliseconds
            success, frame = video.read()
            if success:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if previous_frame_gray is not None:
                    similarity, _ = ssim(previous_frame_gray, frame_gray, full=True)
                    difference = 1 - similarity  # SSIM returns similarity not difference
                    ssim_differences.append(difference)

                previous_frame_gray = frame_gray

                # Save all frames
                all_frames_path = os.path.join(all_frames_dir, f'frame_{ms}.jpg')
                plt.figure()
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                plt.title(f'Frame at {ms} ms')
                plt.axis('off')
                plt.savefig(all_frames_path)
                plt.close()
            else:
                logger.warning(f"Warning: Frame at {ms} ms could not be read.")

        # Calculate dynamic SSIM threshold based on variance in differences
        if ssim_differences:
            mean_diff = np.mean(ssim_differences)
            std_diff = np.std(ssim_differences)
            threshold = mean_diff + std_diff
        else:
            threshold = 0.10  # Default threshold if no SSIM differences calculated

        logger.info(f"Dynamic SSIM threshold set to: {threshold:.2%}")

        previous_frame_gray = None

        for ms in range(interval_ms, duration_ms + 1, interval_ms):
            video.set(cv2.CAP_PROP_POS_MSEC, ms)
            success, frame = video.read()
            if success:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if previous_frame_gray is not None:
                    similarity, _ = ssim(previous_frame_gray, frame_gray, full=True)
                    difference = 1 - similarity

                    if difference > threshold:
                        frames.append(frame)
                        frame_count += 1
                        logger.info(f"Frame at {ms} ms added with SSIM difference ratio: {difference:.2%}")
                        # Save the selected frame
                        frame_path = os.path.join(output_frames_dir, f'frame_{ms}.jpg')
                        plt.figure()
                        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        plt.title(f'Frame at {ms} ms')
                        plt.axis('off')
                        plt.savefig(frame_path)
                        plt.close()

                        if frame_count >= max_frames:
                            break
                    else:
                        logger.info(f"Skipping frame at {ms} ms due to low SSIM difference ratio: {difference:.2%}")

                previous_frame_gray = frame_gray
            else:
                logger.warning(f"Warning: Frame at {ms} ms could not be read.")

        logger.info(f"{frame_count} frames read and selected (every {interval_ms} ms).")

        if not frames:
            raise ValueError("No frames selected for the collage.")

        grid_size = int(np.ceil(np.sqrt(len(frames))))
        frame_height, frame_width, _ = frames[0].shape

        collage_height = grid_size * frame_height
        collage_width = grid_size * frame_width
        collage = np.zeros((collage_height, collage_width, 3), dtype=np.uint8)

        for idx, frame in enumerate(frames):
            row = idx // grid_size
            col = idx % grid_size
            y = row * frame_height
            x = col * frame_width
            collage[y:y + frame_height, x:x + frame_width] = frame

        # Resize the collage to the specified width
        max_width = 768
        scaling_factor = max_width / collage_width
        resized_collage = cv2.resize(collage, (max_width, int(collage_height * scaling_factor)))

        _, buffer = cv2.imencode(".jpg", resized_collage)
        base64_collage = base64.b64encode(buffer).decode("utf-8")

        # Save and display the final collage
        collage_path = os.path.join(output_frames_dir, 'final_collage.jpg')
        logger.info(f'Saving final collage to {collage_path}')
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(resized_collage, cv2.COLOR_BGR2RGB))
        plt.title('Final Collage')
        plt.axis('off')
        plt.savefig(collage_path)
        plt.close()

        return base64_collage
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


async def generate_func_flow(video_path):
    logger.info("inside generate_func_flow")
    global base64_collage
    with open(video_path, "rb") as video_file:
        base64_collage = await capture_frames_at_intervals(video_file, 250)

    # Prepare prompt messages
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": '''I am testing an Android application using video analysis to understand its functionality.
                    Specifically, I need to analyze the video frames of the application to generate a detailed functionality flow based on user interactions, focusing on the static UI elements and predefined states.
                
                    The video frames have been captured and arranged into a collage image. The collage should be read from left to right and top to bottom. Each row of the collage represents a sequence of frames captured at regular intervals.
                
                    Please follow these steps:
                
                    1. **Analyze the Collage:**
                       - Observe the collage image to identify the sequence of user interactions with the application.
                       - Note down each step in detail, including any screen transitions, user inputs, and system responses, focusing on static UI elements and not on dynamic data from API responses.
                
                    2. **Generate the Functional Flow:**
                       - Provide a detailed flow of the feature based on the observed interactions in the collage.
                       - Clearly depict each step, including relevant conditions or branching logic triggered by user interactions or predictable system responses.
                       - Ensure that each step is described in the order it occurs, emphasizing static elements like buttons, input fields, labels, and predefined messages.
                
                    Example Structure:
                
                    **Functional Flow:**
                    - Step 1: [Description of user interaction and initial state, e.g., "User launches the app and observes the splash screen."]
                    - Step 2: [Description of subsequent interaction and app response, focusing on static elements, e.g., "User navigates to the main screen and sees options for Trains, Flights, Buses, and Hotels."]
                    - Step 3: [Repeat for each step observed, describing user interactions and static components only.]
                
                    Focus: Capture the functionality flow based on user interactions as seen in the collage, 
                    concentrating on static UI elements and avoiding reliance on dynamic data. Each step should be clear and concise, 
                    capturing the essence of user actions and predictable app behavior.
                
                    This is a collage of frames from a video that I want to upload.'''
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_collage}"
                    }
                }
            ]
        }
    ]

    params = {
        "model": "gpt-4o-mini",
        "messages": prompt_messages,
        "max_tokens": 4096,
        "temperature": 0.3,  # Lower temperature for more deterministic and precise responses
        "top_p": 0.8,
    }

    result = client.chat.completions.create(**params)
    logger.info("result: %s", result.choices[0].message.content)

    input_tokens = result.usage.prompt_tokens
    output_tokens = result.usage.completion_tokens
    total_tokens = result.usage.total_tokens

    input_cost = (input_tokens / 1_000_000) * input_cost_per_million
    output_cost = (output_tokens / 1_000_000) * output_cost_per_million
    total_cost = input_cost + output_cost

    logger.info(f"Input tokens: {input_tokens}")
    logger.info(f"Output tokens: {output_tokens}")
    logger.info(f"Total tokens: {total_tokens}")
    logger.info(f"Input cost: ${input_cost:.6f}")
    logger.info(f"Output cost: ${output_cost:.6f}")
    logger.info(f"Total cost: ${total_cost:.6f}")

    return result.choices[0].message.content


async def generate_test_cases(video_path: str, application_flow: str):
    logger.info("inside generate_test_cases")
    global base64_collage
    if base64_collage is None:
        with open(video_path, "rb") as video_file:
            base64_collage = await capture_frames_at_intervals(video_file, 250)

    prompt_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f'''Based on the detailed functionality flow generated from the video frames,
                    I need to create comprehensive UI-based test cases based on the detailed functionality flow generated from the video frames.

                    Please make sure that the test cases are comprehensive, covering all possible scenarios as observed in the video frames, and have expected outcomes based on static UI elements in the video frames and not dependent on dynamic data from APIs.
    
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
                        
                    Note: The names of the impacted screens should match the following list exactly: {list(screen_mapper_android.keys())}.
                        
                    These are all the screens that are available: {all_screens}
                    
                    This is a collage of frames from a video that I want to upload.'''
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_collage}"
                    }
                }
            ]
        }
    ]

    params = {
        "model": "gpt-4o-mini",
        "messages": prompt_messages,
        "max_tokens": 4096,
        "temperature": 0.3,  # Lower temperature for more deterministic and precise responses
        "top_p": 0.8,
    }

    result = client.chat.completions.create(**params)
    logger.info("result: %s", result.choices[0].message.content)

    input_tokens = result.usage.prompt_tokens
    output_tokens = result.usage.completion_tokens
    total_tokens = result.usage.total_tokens

    input_cost = (input_tokens / 1_000_000) * input_cost_per_million
    output_cost = (output_tokens / 1_000_000) * output_cost_per_million
    total_cost = input_cost + output_cost

    logger.info(f"Input tokens: {input_tokens}")
    logger.info(f"Output tokens: {output_tokens}")
    logger.info(f"Total tokens: {total_tokens}")
    logger.info(f"Input cost: ${input_cost:.6f}")
    logger.info(f"Output cost: ${output_cost:.6f}")
    logger.info(f"Total cost: ${total_cost:.6f}")

    return result.choices[0].message.content


async def generate_code_for_test_cases(video_path: str,
                                       application_flow: str,
                                       test_cases_list: str,
                                       os_type: str = "android"):
    logger.info("inside generate_code_for_test_cases")
    logger.info("OS: %s", os_type)

    global base64_collage
    if base64_collage is None:
        with open(video_path, "rb") as video_file:
            base64_collage = await capture_frames_at_intervals(video_file, 250)

    impacted_screens = set()
    lines = test_cases_list.split("\n")
    for line in lines:
        if line.startswith("- **Impacted Screens:**"):
            screens = [screen.strip().strip("[]") for screen in line.replace("- **Impacted Screens:**", "").split(",")]
            impacted_screens.update(screens)

    logger.info("impacted_screens: %s", impacted_screens)
    screen_mapper = screen_mapper_android if os_type == "android" else screen_mapper_ios
    screen_data = {screen: screen_mapper.get(screen, {}) for screen in impacted_screens}

    prompt_messages_for_android = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f'''I have developed test cases for an Android application.
                    Using the provided video frames, functional flow, the xpaths of UI elements, and test cases,
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

                        def test_case_1():
                            # Verify that the app launches and displays the home screen with app icons.
                            try:
                                WebDriverWait(driver, 10).until(
                                    EC.presence_of_element_located((AppiumBy.XPATH, "//android.widget.TextView[@text='Trains']"))
                                )
                                assert driver.find_element(AppiumBy.XPATH, "//android.widget.TextView[@text='Trains']").is_displayed()
                                logging.info("Test Case 1 passed: Home screen loaded successfully")
                            except Exception as e:
                                logging.error(f"Test Case 1 failed: {{e}}")
                                raise

                        def test_case_2():
                            # Verify that the PNR status screen displays the correct title and input field.
                            try:
                                WebDriverWait(driver, 10).until(
                                    EC.presence_of_element_located((AppiumBy.XPATH, "//android.widget.TextView[@text='PNR Status']"))
                                )
                                train_status_button = driver.find_element(AppiumBy.XPATH, "//android.widget.TextView[@text='PNR Status']")
                                train_status_button.click()

                                WebDriverWait(driver, 10).until(
                                    EC.presence_of_element_located((AppiumBy.XPATH, "//android.widget.TextView[@text='Running Status']"))
                                )
                                assert driver.find_element(AppiumBy.XPATH, "//android.widget.TextView[@text='Running Status']").is_displayed()
                                assert driver.find_element(AppiumBy.XPATH, "//android.widget.EditText[@text='Enter your 10 digit PNR']").is_displayed()
                                # assert driver.find_element(AppiumBy.XPATH, "//android.widget.Button[@text='Search']").is_displayed()
                                assert driver.find_element(AppiumBy.ID, "com.ixigo.train.ixitrain:id/btn_search").is_displayed()
                                logging.info("Test Case 2 passed: PNR status screen displays correct elements")
                            except Exception as e:
                                logging.error(f"Test Case 2 failed: {{e}}")
                                raise

                    Notes: 
                    - Please use # instead of """ for comments in the code.
                    - Please ensure any test case with a try block has a corresponding except block to handle exceptions.
                    
                    This is a collage of frames from a video that I want to upload.'''
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_collage}"
                    }
                }
            ],
        },
    ]

    prompt_messages_for_ios = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f'''I have developed test cases for an iOS application.
                    Using the provided video frames, functional flow, the xpaths of UI elements, and test cases,
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

                    This is a collage of frames from a video that I want to upload.'''
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_collage}"
                    }
                }
            ],
        },
    ]

    params = {
        "model": "gpt-4o-mini",
        "messages": prompt_messages_for_android if os_type == 'android' else prompt_messages_for_ios,
        "max_tokens": 4096,
        "temperature": 0.3,  # Lower temperature for more deterministic and precise responses
        "top_p": 0.8,
    }

    result = client.chat.completions.create(**params)
    logger.info("result: %s", result.choices[0].message.content)

    input_tokens = result.usage.prompt_tokens
    output_tokens = result.usage.completion_tokens
    total_tokens = result.usage.total_tokens

    input_cost = (input_tokens / 1_000_000) * input_cost_per_million
    output_cost = (output_tokens / 1_000_000) * output_cost_per_million
    total_cost = input_cost + output_cost

    logger.info(f"Input tokens: {input_tokens}")
    logger.info(f"Output tokens: {output_tokens}")
    logger.info(f"Total tokens: {total_tokens}")
    logger.info(f"Input cost: ${input_cost:.6f}")
    logger.info(f"Output cost: ${output_cost:.6f}")
    logger.info(f"Total cost: ${total_cost:.6f}")

    return result.choices[0].message.content


@app.post("/test/")
async def create_test(video: UploadFile = File(...)):
    test_id = str(uuid.uuid4())
    video_path = f"./videos/{test_id}.mp4"

    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    database["tests"][test_id] = {
        "video_path": video_path,
    }

    return {
        "status": True,
        "data": {
            "testId": test_id,
        }
    }


@app.post("/func_flow/{test_id}")
async def create_func_flow(test_id: str):
    video_path = f"./videos/{test_id}.mp4"

    func_flow = await generate_func_flow(video_path)

    database["func_flows"][test_id] = func_flow

    return {
        "status": True,
        "data": {
            "funcFlow": func_flow
        }
    }


def get_func_flow(test_id: str):
    return "**Functional Flow:**\n\n- **Step 1:** User launches the app and observes the splash screen with the app logo and loading animation.\n\n- **Step 2:** User navigates to the main screen, which displays options for \"Trains,\" \"Flights,\" \"Buses,\" and \"Hotels,\" along with a search bar at the top.\n\n- **Step 3:** User selects the \"Trains\" option, which leads to a new screen showing the \"Running Status\" feature.\n\n- **Step 4:** User is prompted to \"Enter your 10 digit PNR\" in the search bar. The search button is visible next to the input field.\n\n- **Step 5:** User inputs a 10-digit PNR number and taps the \"Search\" button.\n\n- **Step 6:** The app processes the input and displays a message indicating \"PNR No. is not valid\" if the input is incorrect.\n\n- **Step 7:** User corrects the PNR input and taps the \"Search\" button again.\n\n- **Step 8:** Upon valid input, the app displays the train details, including train name, departure time, and arrival time.\n\n- **Step 9:** User views additional options such as \"Platform Locator,\" \"Refund Calculator,\" and promotional offers for credit cards.\n\n- **Step 10:** User scrolls down to see more options, including advertisements for other apps and services.\n\n- **Step 11:** User selects a specific train to view more details, including passenger information and train ratings.\n\n- **Step 12:** User can see the train's route and additional information about the train's performance and reviews.\n\nThis flow captures the sequence of user interactions and the static UI elements present in the collage, focusing on the app's functionality without referencing dynamic data."


@app.post("/test_cases/{test_id}")
async def create_test_cases(test_id: str):
    video_path = f"./videos/{test_id}.mp4"
    func_flow = get_func_flow(test_id=test_id)

    test_cases = await generate_test_cases(video_path, func_flow)

    database["test_cases"][test_id] = test_cases

    return {
        "status": True,
        "data": {
            "testCases": test_cases
        }
    }


def get_test_cases(test_id: str):
    return "Here are the comprehensive UI-based test cases based on the provided functional flow:\n\n### Test Case 1:\n- **Description:** Verify the splash screen displays the app logo and loading animation upon app launch.\n- **Impacted Screens:** home_screen\n- **Steps:**\n  1. Launch the app.\n  2. Observe the splash screen.\n- **Expected Outcome:** The splash screen should display the app logo and a loading animation.\n\n### Test Case 2:\n- **Description:** Verify navigation to the main screen after the splash screen.\n- **Impacted Screens:** home_screen\n- **Steps:**\n  1. Launch the app.\n  2. Wait for the splash screen to transition to the main screen.\n- **Expected Outcome:** The main screen should display options for \"Trains,\" \"Flights,\" \"Buses,\" and \"Hotels,\" along with a search bar at the top.\n\n### Test Case 3:\n- **Description:** Verify the \"Trains\" option navigates to the PNR status screen.\n- **Impacted Screens:** home_screen, pnr_status_screen\n- **Steps:**\n  1. On the main screen, select the \"Trains\" option.\n- **Expected Outcome:** The app should navigate to the PNR status screen, displaying the \"Enter your 10 digit PNR\" prompt and the search button.\n\n### Test Case 4:\n- **Description:** Verify the input field for entering the PNR number is present and functional.\n- **Impacted Screens:** pnr_status_screen\n- **Steps:**\n  1. Navigate to the PNR status screen.\n  2. Check for the presence of the input field labeled \"Enter your 10 digit PNR.\"\n- **Expected Outcome:** The input field should be visible and ready for user input.\n\n### Test Case 5:\n- **Description:** Verify the search button is functional when a valid PNR number is entered.\n- **Impacted Screens:** pnr_status_screen\n- **Steps:**\n  1. Enter a valid 10-digit PNR number into the input field.\n  2. Tap the \"Search\" button.\n- **Expected Outcome:** The app should process the input and display the train details, including train name, departure time, and arrival time.\n\n### Test Case 6:\n- **Description:** Verify the app displays an error message for an invalid PNR number.\n- **Impacted Screens:** pnr_status_screen\n- **Steps:**\n  1. Enter an invalid PNR number (less than 10 digits or non-numeric).\n  2. Tap the \"Search\" button.\n- **Expected Outcome:** The app should display a message indicating \"PNR No. is not valid.\"\n\n### Test Case 7:\n- **Description:** Verify the app processes a corrected valid PNR number.\n- **Impacted Screens:** pnr_status_screen\n- **Steps:**\n  1. Enter an invalid PNR number and tap the \"Search\" button.\n  2. Correct the PNR number to a valid 10-digit number.\n  3. Tap the \"Search\" button again.\n- **Expected Outcome:** The app should display the train details correctly.\n\n### Test Case 8:\n- **Description:** Verify additional options are displayed after viewing train details.\n- **Impacted Screens:** pnr_status_screen\n- **Steps:**\n  1. After viewing train details, scroll down the screen.\n- **Expected Outcome:** The app should display options such as \"Platform Locator,\" \"Refund Calculator,\" and promotional offers for credit cards.\n\n### Test Case 9:\n- **Description:** Verify scrolling functionality to view advertisements and additional options.\n- **Impacted Screens:** pnr_status_screen\n- **Steps:**\n  1. Scroll down the PNR status screen.\n- **Expected Outcome:** The app should allow scrolling, revealing advertisements for other apps and services.\n\n### Test Case 10:\n- **Description:** Verify selecting a specific train displays detailed information.\n- **Impacted Screens:** pnr_status_screen\n- **Steps:**\n  1. Select a specific train from the displayed list.\n- **Expected Outcome:** The app should display detailed information about the train, including passenger information and train ratings.\n\n### Test Case 11:\n- **Description:** Verify viewing the train's route and performance information.\n- **Impacted Screens:** pnr_status_screen\n- **Steps:**\n  1. After selecting a specific train, look for route and performance information.\n- **Expected Outcome:** The app should display the train's route and additional information about performance and reviews.\n\nThese test cases cover the interactions and expected outcomes based on the static UI elements observed in the video frames, ensuring comprehensive testing of the application."


@app.post("/test_cases/code/{test_id}")
async def create_test_cases_code(test_id: str):
    video_path = f"./videos/{test_id}.mp4"
    func_flow = get_func_flow(test_id=test_id)
    test_cases = get_test_cases(test_id=test_id)

    test_cases_code = await generate_code_for_test_cases(video_path, func_flow, test_cases)

    database["test_cases_code"][test_id] = test_cases_code

    return {
        "status": True,
        "data": {
            "testCases": test_cases_code
        }
    }


def get_test_cases_code(test_id: str):
    return "Here's the Appium code to automate the testing of the specified functionality in the Android application based on the provided test cases and functional flow. Each test case includes appropriate assertions and comments.\n\n```python\nfrom appium import webdriver\nfrom appium.webdriver.common.appiumby import AppiumBy\nfrom selenium.webdriver.support.ui import WebDriverWait\nfrom selenium.webdriver.support import expected_conditions as EC\nimport logging\n\n# Initialize the Appium driver (ensure to set desired capabilities)\ndriver = webdriver.Remote('http://localhost:4723/wd/hub', desired_capabilities)\n\ndef test_case_1():\n    # Verify that the app launches and displays the home screen with app icons.\n    try:\n        WebDriverWait(driver, 10).until(\n            EC.presence_of_element_located((AppiumBy.XPATH, \"//android.widget.TextView[@text='Trains']\"))\n        )\n        assert driver.find_element(AppiumBy.XPATH, \"//android.widget.TextView[@text='Trains']\").is_displayed()\n        logging.info(\"Test Case 1 passed: Home screen loaded successfully\")\n    except Exception as e:\n        logging.error(f\"Test Case 1 failed: {e}\")\n        raise\n\n# Execute test cases\ntest_case_1()\n\n# Close the driver after tests\ndriver.quit()\n```\n\n### Notes:\n- Each test case includes a try-except block to handle exceptions and log the results.\n- The code assumes that the Appium server is running and the desired capabilities are set correctly for the Android application.\n- Adjust the XPaths and element identifiers as necessary based on the actual UI elements in the application."


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
    Extract test case functions from the provided code snippet and append their calls.
    
    :param code_snippet: The entire code snippet as a string.
    :return: A string containing all test case functions and their calls.
    """
    # Regular expression to match class methods starting with 'test_case_' and include all indented lines
    test_case_pattern = re.compile(r'(def test_case_\d+\(\):\n(?: {4}.*\n)*)', re.MULTILINE)

    # Find all matches in the provided code snippet
    matches = test_case_pattern.findall(code_snippet)

    # Join all matches with a newline to form the output string
    result = "\n".join(matches)

    # Extract test case function names
    test_case_calls = [re.search(r'def (test_case_\d+)\(\):', match).group(1) for match in matches]

    # Append the test case calls to the result
    result += "\n\n" + "\n".join(f"{test_case_call}()" for test_case_call in test_case_calls)

    return result


@app.post("/test/execute/{test_id}")
async def execute_test(test_id: str):
    test_cases_code = get_test_cases_code(test_id=test_id)
    extracted_test_cases = extract_test_cases(test_cases_code)

    results = []

    try:
        setup()
        for test_case in extracted_test_cases.split("\n\n"):
            try:
                print(test_case)
                exec(test_case)
                if test_case.strip().startswith("def "):
                    test_case_id = re.search(r'def (test_case_\d+)\(\):', test_case).group(1)
                    test_case_description = test_case.splitlines()[1].strip().lstrip("# ") if len(test_case.splitlines()) > 1 else ""
                    results.append({
                        "testId": test_id,
                        "testCaseId": test_case_id,
                        "testCaseDescription": test_case_description,
                        "status": "PASSED"
                    })
            except Exception as e:
                if test_case.strip().startswith("def "):
                    test_case_id = re.search(r'def (test_case_\d+)\(\):', test_case).group(1)
                    test_case_description = test_case.splitlines()[1].strip().lstrip("# ")
                    results.append({
                        "testId": test_id,
                        "testCaseId": test_case_id,
                        "testCaseDescription": test_case_description,
                        "status": "FAILED"
                    })
                    logging.error(f"{test_case_id} failed: {e}")
    except Exception as e:
        logging.error(f"Error during test execution: {e}")
    finally:
        teardown()

    return {
        "status": True,
        "data": {
            "results": results
        }
    }
