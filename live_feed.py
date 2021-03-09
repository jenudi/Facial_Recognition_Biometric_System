from live_feed_utils import *


if __name__ == "__main__":

    faces_detected_dir='\\'.join([os.getcwd(),'faces detected in live feed'])
    if not os.path.isdir(faces_detected_dir):
        os.mkdir(faces_detected_dir)

    no_faces_detected_dir='\\'.join([os.getcwd(),'no faces detected in live feed'])
    if not os.path.isdir(no_faces_detected_dir):
        os.mkdir(no_faces_detected_dir)


    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    face_detected_number=0
    no_face_detected_number=0
    while True:
        ret, frame = cap.read()
        frame = Captured_frame(cv.resize(frame, None, fx=0.5, fy=0.5,interpolation=cv.INTER_AREA))
        frame.set_face_image()
        if frame.face_detected:
            print("face detected")
            frame.face_image.resize_image()
            face_detected_number+=1
            frame.face_image.save("".join([faces_detected_dir,"\\",str(face_detected_number),".jpg"]))
            frame.identify("normalize_by_train_values",train_paths,id_to_name_dict)
            if frame.face_recognized:
                now = datetime.now().strftime('%Y %m %d %H %M %S').split(' ')
                register_entry(frame.id_detected,now,attendance_collection)
        else:
            print("no face detected")
            no_face_detected_number+=1
            frame.save("".join([no_faces_detected_dir,"\\",str(no_face_detected_number),".jpg"]))
        c = cv.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv.destroyAllWindows()