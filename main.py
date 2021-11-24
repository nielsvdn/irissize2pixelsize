import cv2
import mediapipe as mp
import math as m
import numpy as np

# https://towardsdatascience.com/face-landmark-detection-using-python-1964cb620837


def distance_between_points(loc1, loc2):
    """
    Calculate the distance between two sets of coordinates.

    Parameters
    ----------
    loc1 : tuple
        Coordinates of first location.
    loc2 : tuple
        Coordinates of second location.

    Returns
    -------
    float
        The distance between the two locations.

    """
    dx = abs(loc1[0] - loc2[0])
    dy = abs(loc1[1] - loc2[1])

    return m.sqrt(dx ** 2 + dy ** 2)


def draw_double_sided_arrow_with_text(img, loc1, loc2, text):
    """
    Draw a double sides arrow with a text annotation on a image.

    Parameters
    ----------
    img : Array of uint8
        An image array.
    loc1 : tuple
        Coordinates of first end of the arrow.
    loc2 : tuple
        Coordinates of second end of the arrow.
    text : string
        The text annotation.

    Returns
    -------
    img : Array of uint8
        The image array with the arrow and annotation.

    """
    # Define color
    color = (0, 0, 255)

    # Draw double sides arrow
    img = cv2.arrowedLine(img, loc1, loc2, color, 5)
    img = cv2.arrowedLine(img, loc2, loc1, color, 5)

    # Calculate coordinates in the middle of the arrow
    dx = finger_locations[0][0] - finger_locations[1][0]
    dy = finger_locations[0][1] - finger_locations[1][1]
    x = int(finger_locations[0][0] - dx / 2)
    y = int(finger_locations[0][1] - dy / 2)

    # Draw annotation
    img = cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)

    return img


class iris_pixel_size_detector:
    """ Detect the pixel size by analysing the iris. """

    def __init__(self, img,):
        # Define properties of iris
        self.landmarks_eyes = {
            "left": [474, 475, 476, 477],
            "right": [469, 470, 471, 472],
        }
        self.opposites = [(0, 2), (1, 3), (4, 6), (5, 7)]
        self.iris_size_mm = 11.7  # +- 0.5

        # Get size of image
        self.w, self.h = img.shape[1], img.shape[0]

        # Define iris detector
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def get_pixel_size(self, img):
        """
        Calculate the pixel size from an given image with a face.

        Parameters
        ----------
        img : Array of uint8
            An image with a face.

        Returns
        -------
        float
            The size of the pixels in mm.

        """
        # detect face mesh
        results = self.face_mesh.process(img)

        # Check if face is found
        if results.multi_face_landmarks:

            # Select first face of all detected faces
            face_landmarks = results.multi_face_landmarks[0]

            # Get positions of landmarks
            locs = []
            for landmarks in self.landmarks_eyes.values():
                for landmark in landmarks:
                    # Get landmark position
                    x = face_landmarks.landmark[landmark].x * self.w
                    y = face_landmarks.landmark[landmark].y * self.h

                    # Save position
                    locs.append((x, y))

            # Calculate distance between points opposite to each other
            distances = []
            for opposite in self.opposites:
                loc1 = locs[opposite[0]]
                loc2 = locs[opposite[1]]
                dist = distance_between_points(loc1, loc2)
                distances.append(dist)
            iris_size_pixel = np.mean(distances)

            # Return pixel size
            return self.iris_size_mm / iris_size_pixel

        else:
            # No face detected, return none
            return np.nan


class finger_detector:
    """Detect the indexfinger and thumb."""

    def __init__(self, img):
        # Define hand detector
        mphands = mp.solutions.hands
        self.hands = mphands.Hands(max_num_hands=1)
        self.landmarks = [4, 8]

        # Get size of image
        self.w, self.h = img.shape[1], img.shape[0]

    def get_locations(self, img):
        """
        Determine the location of the thumb and index fingers.

        Parameters
        ----------
        img : Array of uint8
            An image with a hand.

        Returns
        -------
        locs : list
            An list filled with coordinates of the thumb and index fingers.

        """
        # detect hand
        results = self.hands.process(img)

        # Check if hand is found
        locs = []
        if results.multi_hand_landmarks:

            # Select first hand of all detected hands
            hand_landmarks = results.multi_hand_landmarks[0]

            # Save position of landmarks
            for landmark in self.landmarks:
                x = hand_landmarks.landmark[landmark].x * self.w
                y = hand_landmarks.landmark[landmark].y * self.h
                locs.append((int(x), int(y)))

        return locs


# set and test capture device
cap = cv2.VideoCapture(0)
success, img = cap.read()
if not success:
    raise OSError("Can't take picture, webcam probably not available.")

# Initialise pixel size detector
psd = iris_pixel_size_detector(img)

# Initialise finger detector
fd = finger_detector(img)

while cap.isOpened():
    # Capture image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Convert image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Calculate pixel size
    pixel_size = psd.get_pixel_size(img)

    # Get location of fingers
    finger_locations = fd.get_locations(img)

    # Draw arrow and distance
    if (len(finger_locations) > 1) and not (np.isnan(pixel_size)):
        # Calculate distance
        dist_px = distance_between_points(finger_locations[0], finger_locations[1])
        dist_mm = dist_px * pixel_size

        # Make correction for finger thickness
        dist_mm -= 10  # mm

        # Draw arrow
        text = str(int(round(dist_mm / 10, 0))) + "cm"
        img = draw_double_sided_arrow_with_text(
            img, finger_locations[0], finger_locations[1], text
        )

    # Convert image to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Show image
    cv2.imshow("Size calculator", img)

    # Check if user pressed escape
    if cv2.waitKey(5) & 0xFF == 27:
        cap.release()
        cv2.destroyAllWindows()
        raise KeyboardInterrupt("User escaped program")
