import tkinter as tk
import cv2 as cv
import numpy as np
from PIL import Image, ImageTk
import threading  # Import the threading module

# Function to perform motion detection on the video stream and control the AC
def detect_motion_and_control_ac():
    try:
        # Open video capture object
        capture = cv.VideoCapture(0)  # 0 for webcam
        
        # Check if the video capture object is opened successfully
        if not capture.isOpened():
            raise Exception("Unable to open video source.")
        
        # Read the first frame
        ret, prev_frame = capture.read()
        
        # Check if the first frame is read successfully
        if not ret:
            raise Exception("Unable to read the first frame.")
        
        # Convert frame to grayscale
        prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
        
        # Initialize AC status
        ac_status = "OFF"
        
        while True:
            # Read the next frame
            ret, frame = capture.read()
            
            # Check if the frame is read successfully
            if not ret:
                print("End of video stream.")
                break
            
            # Convert frame to grayscale
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            # Compute the absolute difference between the current and previous frame
            frame_diff = cv.absdiff(gray, prev_gray)
            
            # Apply a threshold to the difference to detect motion
            _, thresh = cv.threshold(frame_diff, 30, 255, cv.THRESH_BINARY)
            
            # Find contours in the thresholded image
            contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            # Draw rectangles around the detected motion
            for contour in contours:
                if cv.contourArea(contour) > 500:
                    x, y, w, h = cv.boundingRect(contour)
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Update AC status to ON if human detected
                    ac_status = "ON"
            
            # Convert the frame to RGB format
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            # Convert the frame to ImageTk format
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            
            # Display the frame with motion detection
            label.config(image=img_tk)
            label.image = img_tk
            
            # Update the previous frame
            prev_gray = gray.copy()
            
            # Check for exit key press
            if cv.waitKey(1) & 0xFF == ord('q'):
                print("Motion detection stopped by user.")
                break
        
        # Turn off the AC if no human detected
        if ac_status == "OFF":
            print("No human detected. Turning off AC.")
        else:
            print("Human detected. Keeping AC on.")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Release video capture object and close window
        if 'capture' in locals() and capture.isOpened():
            capture.release()
        cv.destroyAllWindows()

# Create Tkinter window
window = tk.Tk()
window.title("Real-Time Motion Detection and AC Control")

# Create label to display video stream
label = tk.Label(window)
label.pack()

# Start motion detection and AC control in a separate thread
motion_detection_thread = threading.Thread(target=detect_motion_and_control_ac)
motion_detection_thread.start()

# Start the Tkinter event loop
window.mainloop()
