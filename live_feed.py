from live_feed_utils import *


if __name__ == "__main__":

    faces_detected_dir='\\'.join([os.getcwd(),'faces detected in live feed'])
    if not os.path.isdir(faces_detected_dir):
        os.mkdir(faces_detected_dir)

    no_faces_detected_dir='\\'.join([os.getcwd(),'no faces detected in live feed'])
    if not os.path.isdir(no_faces_detected_dir):
        os.mkdir(no_faces_detected_dir)


    cap = cv.VideoCapture(0,cv.CAP_DSHOW)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    face_detected_number=0
    no_face_detected_number=0
    while True:
        ret, frame = cap.read()
        if (frame is  None) or (isinstance(frame, type(None))) :
            continue
        frame_image = Captured_frame(cv.resize(frame, None, fx=0.5, fy=0.5,interpolation=cv.INTER_AREA))
        frame_image.set_face_image()
        if frame_image.face_detected:
            print("face detected")
            frame_image.face_image.resize_image()
            face_detected_number+=1
            frame_image.face_image.save("".join([faces_detected_dir,"\\",str(face_detected_number),".jpg"]))
            frame_image.identify("normalize_by_train_values",train_paths,id_to_name_dict)
            if frame_image.face_recognized:
                now = datetime.now().strftime('%Y %m %d %H %M %S').split(' ')
                register_entry(frame_image.id_detected,now,attendance_collection)
        else:
            print("no face detected")
            no_face_detected_number+=1
            frame_image.save("".join([no_faces_detected_dir,"\\",str(no_face_detected_number),".jpg"]))
        c = cv.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv.destroyAllWindows()