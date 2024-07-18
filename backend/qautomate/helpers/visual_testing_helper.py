import os
from openai import OpenAI

# Read the value of the environment variable
openai_key = os.getenv('openai_key')
if not openai_key:
    raise ValueError("No OpenAI API key found in environment variable 'openai_key'")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", openai_key))


# # Function to encode the image
# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')


# Function to analyze image using client
def analyze_image_color(base64_image, model="gpt-4o", max_tokens=4096):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": '''
                        Act as an expert QA in visual testing for Android applications. Your task is to analyze an image of an Android application's interface to inspect the colors used across different UI elements.

                        Please provide a detailed report that includes:
                        1. The colors used for each UI element.

                        I will share the image of the application. Please analyze it carefully and describe the color details and overall color scheme.
                        
                        The image is shared.'''
                     },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def difference_in_colors_analyze(description_image_1, description_image_2, base64_image_1, base64_image_2):
    response_of_color = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
                    Act as an expert QA in visual testing for Android applications. Your task is to compare two images of an Android application's interface. 
                    My goal is to identify differences in two images.
                    I will share the image descriptions and corresponding images and you give me all differences between them. 

                    Please do it carefully as it is critical.

                    Description of Original Image: {description_image_1}
                    Description of Testing Image: {description_image_2}

                    Example structure:

                    #### 1. **Navigation Bar Icons (Top Menu)**
                    - **Original Image:**
                      - **Trains Icon**: Blue with white and black accents.
                      - **Flights Icon**: Blue and white.
                      - **Buses Icon**: Blue and white.
                      - **Hotels Icon**: Blue, white, and orange.
                    - **Testing Image:**
                      - **Trains Icon**: Blue with white accents.
                      - **Flights Icon**: Blue with white accents.
                      - **Hotels Icon**: Blue with white accents.
                      - **Buses Icon**: Blue with white accents.

                    #### 2. **Date Selection:**
                    - **Original Image:**
                      - **Background (Date Section)**: White (#FFFFFF)
                      - **Date Text**: Black (#000000)
                      - **Close Button (X)**: Black (#000000)
                      - **Tomorrow Button**: Green background (#228B22) with white text.
                      - **Day After Button**: Green background (#228B22) with white text.
                    - **Testing:**
                      - **Background (Date Section)**: White (#FFFFFF)
                      - **Date Text**: Black (#000000)
                      - **Close Button (X)**: Black (#000000)
                      - **Day After Button**: Green background (#7FC242) with white text.
                      - **Day After Button**: Green background (#7FC242) with white text.
              """
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{base64_image_1}",
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{base64_image_2}",
                        }
                    }
                ]
            }
        ],
        max_tokens=4096,
        temperature=0.3,  # Lower temperature for more deterministic and precise responses
        top_p=0.5
    )
    return response_of_color.choices[0].message.content


def process_color_in_images(original_image, testing_image):
    print("processing color in images")

    encoded_images = [original_image, testing_image]

    descriptions = []
    for image in encoded_images:
        description = analyze_image_color(image)
        descriptions.append(description)

    # Description for first image
    description_image_1_color = descriptions[0]

    # Description for second image
    description_image_2_color = descriptions[1]

    # difference in image based on color
    color_difference = difference_in_colors_analyze(description_image_1_color, description_image_2_color,
                                                    encoded_images[0], encoded_images[1])
    return color_difference


def analyze_image_layout(base64_image, model="gpt-4o", max_tokens=4096):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": '''
                     Act as an expert QA in visual testing for Android applications. Your task is to analyze an image of an Android application's interface to understand the layout of its UI elements.

                    Please provide a detailed report that includes:
                    1. The location of each UI element within the image.

                    I will share the image of the application. Please analyze it thoroughly and describe the layout comprehensively.
                    Dont share any color related information but only elements layout related analysis.
                        
                    The image is shared.'''
                     },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        max_tokens=max_tokens,
        temperature=0.3,  # Lower temperature for more deterministic and precise responses
        top_p=0.6
    )
    return response.choices[0].message.content


def difference_in_layout_analyze(description_image_1, description_image_2, base64_image_1, base64_image_2):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
                Act as an expert QA in visual testing for Android applications. Your task is to compare two images of an Android application's interface. 
                My goal is to identify differences in two images.
                I will share the image descriptions and corresponding images and you give me all differences between them. 

                Please do it carefully as it is critical.

                Description of Original Image: {description_image_1}
                Description of Testing Image: {description_image_2}
                """
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{base64_image_1}",
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{base64_image_2}",
                        }
                    }
                ]
            }
        ],
        max_tokens=4096,
        temperature=0.2,  # Lower temperature for more deterministic and precise responses
        top_p=0.5
    )
    return response.choices[0].message.content


def process_layout_in_images(original_image, testing_image):
    print("processing layout of images")

    descriptions = []
    encoded_images = [original_image, testing_image]
    for image in encoded_images:
        description = analyze_image_layout(image)
        descriptions.append(description)

    # Description for first image
    description_image_1_layout = descriptions[0]

    # Description for second image
    description_image_2_layout = descriptions[1]

    # Difference in image based on layout
    layout_difference = difference_in_layout_analyze(description_image_1_layout, description_image_2_layout,
                                                     encoded_images[0], encoded_images[1])
    return layout_difference


# Function to analyze image text
def analyze_image_text(base64_image, model="gpt-4o", max_tokens=4096):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": '''
                        Act as an expert QA in visual testing for Android applications. Your task is to analyze an image of an Android application's interface to identify and describe its UI elements and their associated text. 

                        Please provide a detailed report that includes:
                        1. A list of all UI elements present in the image.
                        2. The associated text with each UI element.

                        I will share the image of the application. Please analyze it carefully and provide accurate details.
                        Dont give information about color or layout but specifically about the UI element and text in UI element

                        Image is shared'''
                     },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        max_tokens=max_tokens,
        temperature=0.3,  # Lower temperature for more deterministic and precise responses
        top_p=0.6
    )
    return response.choices[0].message.content


def difference_in_text_analyze(description_image_1, description_image_2, base64_image_1, base64_image_2):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
                Act as an expert QA in visual testing for Android applications. Your task is to compare two images of an Android application's interface. 
                My goal is to identify differences in two images.
                I will share the image descriptions and corresponding images and you give me all differences between them. 

                Please do it carefully as it is critical.

            Description of Original Image: {description_image_1}
            Description of Testing Image: {description_image_2}

            Example structure:

            #### 1. **Navigation Bar Icons (Top Menu)**
            - **Original Image:**
            - **Testing Image:**

            #### 2. **Date Selection:**
            - **Original Image:**
            - **Testing Image:**
            """
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{base64_image_1}",
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{base64_image_2}",
                        }
                    }
                ]
            }
        ],
        max_tokens=4096,
        temperature=0.2,  # Lower temperature for more deterministic and precise responses
        top_p=0.5
    )
    return response.choices[0].message.content


def process_text_in_images(original_image, testing_image):
    print("processing text of images")

    descriptions = []
    encoded_images = [original_image, testing_image]
    for image in encoded_images:
        description = analyze_image_text(image)
        descriptions.append(description)

    # Description for first image
    description_image_1_text = descriptions[0]

    # Description for second image
    description_image_2_text = descriptions[1]

    # Difference in image based on text
    text_difference = difference_in_text_analyze(description_image_1_text, description_image_2_text, encoded_images[0],
                                                 encoded_images[1])
    return text_difference


def visual_analyze(difference_in_text, difference_in_layout, difference_in_colors):
    print("processing final Visual testing results of images")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
                    Act as an expert QA in visual testing for Android applications. 

                    I want a final description based on understanding all the three description.
                    The final description should be based on understanding of all three and it should be brief but
                    should not miss out on any info.
                    
                    Color Differences: {difference_in_colors}
                    Text Differences: {difference_in_text}
                    Layout Differences: {difference_in_layout}

                    Synthesize the descriptions into a single, concise summary. 
                    Please provide the individual descriptions for each of the three areas 
                    (color, layout, UI element and text)  Strictly follow the output structure 

                    Output structure:
                    
                    Summary:
                  
                    ### Top Navigation Bar:
                    Original Image: Features icons for Train, Flight, Bus, and Hotel, with promotional text "UPTO 30% OFF."
                    Testing Image: Similar icons but shuffled order and additional promotional texts like "Save ₹500" and "Min. 15% off."
                    /n/n
                    ### Search Section:
                    Original Image: Displays black text on a white background for station names NDLS - New Delhi and LKO - Lucknow Nr, with labels such as “Wednesday, 19 Jun”, and a bright blue "Search Trains" button.
                    Testing Image: Uses black text on a yellow background for station names NDLS and PNBE - Patna Jn, with labels such as “Monday, 13 Sep” and a "Search Flights" button.
                    /n/n"""
                    }
                ]
            }
        ],
        max_tokens=4096,
        temperature=0.4,  # Lower temperature for more deterministic and precise responses
        top_p=0.6
    )
    return response.choices[0].message.content
