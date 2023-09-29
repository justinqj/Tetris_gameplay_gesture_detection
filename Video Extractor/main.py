# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import os

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

def extract_video(input_dir, filename, output_dir, count):
    filepath = os.path.join(input_dir, filename)
    if os.path.isfile(filepath):
        # Check if file is a video
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.mp4' or ext == '.avi':
            # Open the video file
            video = cv2.VideoCapture(filepath)

            # Set frame counter to 0
            frame_count = 0

            # Loop through the video frames
            while True:
                # Read the next frame
                ret, frame = video.read()

                # If there are no more frames, break out of the loop
                if not ret:
                    break

                # Save the frame as an image file
                output_filename = f'image_{count}.jpg'
                output_dirpath = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_dirpath, frame)

                # Increment the frame counter
                frame_count += 1
                count += 1

            # Release the video file
            video.release()
    else:
        print("File Not Found: {}".format(filepath))

    return count

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Set input and output directories
    # input_dir = '/Users/madjew/Library/CloudStorage/OneDrive-UTS/UTS/2023/Autumn/42028 Deep Learning and CNN/Assignment/A3/Dataset/Videos'
    # output_dir = '/Users/madjew/Library/CloudStorage/OneDrive-UTS/UTS/2023/Autumn/42028 Deep Learning and CNN/Assignment/A3/Dataset/Images/Classes/LeftTwist/'

    input_dir = 'C:/Users/NickM/OneDrive - UTS/UTS/2023/Autumn/42028 Deep Learning and CNN/Assignment/A3/Dataset/Videos'
    output_dir = 'C:/Users/NickM/OneDrive - UTS/UTS/2023/Autumn/42028 Deep Learning and CNN/Assignment/A3/Dataset/Images/Classes/RightTwist/'

    count = 0
    count += extract_video(input_dir, "RightTwist.mp4", output_dir, count)
    # # Loop through all files in the input directory
    # for filename in os.listdir(input_dir):
    #     count += extract_video(input_dir, filename, output_dir, count)
    print(count)
