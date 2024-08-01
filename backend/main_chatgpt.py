import base64
import json
import logging
import os
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from skimage.metrics import structural_similarity as ssim

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


async def capture_frames_at_intervals(video_file, interval_ms=250, max_frames=25):
    global base64_collage
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await video_file.read())
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
        all_frames_dir = os.path.abspath('/app/backend/all_frames')
        if not os.path.exists(all_frames_dir):
            os.makedirs(all_frames_dir)

        output_frames_dir = os.path.abspath('/app/backend/output_frames')
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


@app.post("/func_flow/")
async def generate_func_flow(file: UploadFile = File(...)):
    logger.info("inside generate_func_flow")
    global base64_collage
    base64_collage = await capture_frames_at_intervals(file, 250)

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

    return {"result": result.choices[0].message.content}


@app.post("/generate_test_cases/")
async def generate_test_cases(file: UploadFile = File(...),
                              application_flow: str = Form(...)
                              ):
    logger.info("inside generate_test_cases")
    global base64_collage
    if base64_collage is None:
        base64_collage = await capture_frames_at_intervals(file, 250)

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

    return {"result": result.choices[0].message.content}


@app.post("/generate_test_cases_code")
async def generate_code_for_test_cases(file: UploadFile = File(...),
                                       application_flow: str = Form(...),
                                       test_cases_list: str = Form(...),
                                       os_type: str = Form(...),
                                       ):
    logger.info("inside generate_code_for_test_cases")
    logger.info("OS: %s", os_type)

    test_case_list_obj = json.loads(test_cases_list)

    global base64_collage
    if base64_collage is None:
        base64_collage = await capture_frames_at_intervals(file, 250)

    impacted_screens = set()
    lines = test_case_list_obj.split("\n")
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
                            """
                            Verify that the app launches and displays the home screen with app icons.
                            """
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
                            """
                            Verify that the PNR status screen displays the correct title and input field.
                            """
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

    return {"result": result.choices[0].message.content}
