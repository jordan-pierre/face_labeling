import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('face_labeling/haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('face_labeling/test-portrait.jpg')

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

label_color = (36, 255, 12)
# blue: (255, 0, 0)
# white: (255, 255, 255)
# green: (36, 255, 12)
# black: (0, 0, 0)
label_width = 10
label_height = 40
half_label_height = int(label_height * 0.25)
half_label_width = int(label_width * 0.5)

label_text = 'Jordan Pierre'
label_text_color = (0, 0, 0)
label_text_width = 3


# Draw rectangle around the faces
for (x, y, w, h) in faces:
    # Rectangle around face
    cv2.rectangle(img, (x, y), (x + w, y + h), label_color, label_width)
    # Rectangle around text
    cv2.rectangle(img, (x - half_label_width, y - label_height), (x + w + half_label_width, y), label_color, -1)
    # Text above face
    cv2.putText(img, label_text, (x + half_label_width, y - half_label_height), cv2.FONT_HERSHEY_SIMPLEX, 1, label_text_color, label_text_width)

# Display the output
cv2.imshow('window', img)
cv2.waitKey()

# Save file
cv2.imwrite('face_labeling/labeled-portrait.jpg', img)
