import face_recognition
import os
import cv2


KNOWN_FACES_DIR = 'Known_faces'
UNKNOWN_FACES_DIR ='unknow_faces'
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'  
print('Loading known faces...')
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):

    # Now we iterate over our known faces directory, 
    # which contains possibly many directories of identities, 
    # which then contain one or more images with that person's face.
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):

        # Load an image
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only 
        # (assuming one face per image as you can't be twice on one image)
        encoding = face_recognition.face_encodings(image)[0]

        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(name)
print('Processing unknown faces...')
# Now let's loop over a folder of faces we want to label
for filename in os.listdir(UNKNOWN_FACES_DIR):

    # Load image
    print(f'Filename {filename}', end='')
    image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}') 
   # While known_images are just face shots,
   #  we assume that unknown images might have multiple people and other objects in them. Thus, 
   # we want to first locate those faces
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    print(f', found {len(encodings)} face(s)')
    for face_encoding, face_location in zip(encodings, locations):
        # We use compare_faces (but might use face_distance as well)
        # Returns array of True/False values in order of passed known_faces

        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        #we passed our image through face_locations and face_encodings, so we can modify it
    # First we need to convert it from RGB to BGR as we are going to work with cv2
        def name_to_color(name):
            #Now we want to draw a rectangle around this recognized face. 
            # To draw a rectangle in OpenCV, we need the top left and bottom right coordinates, 
            # and we use cv2.rectangle to draw it
    
            color = [(ord(c.lower())-97)*8 for c in name[:3]]
            return color
        match = None
        if True in results: 
            match = known_names[results.index(True)]
            print(f' - {match} from {results}')

    
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

        
            color = name_to_color(match)

   
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

           
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            # Wite a name
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)
            cv2.imshow(filename, image)
            cv2.waitKey(0)
            cv2.destroyWindow(filename) 
     

#match = known_names[results.index(True)]
#print(match)
#color = [(ord(c.lower())-97)*8 for c in match[:3]]
#print(color)


    


     