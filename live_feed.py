from live_feed_utils import *


if __name__ == "__main__":

    live_feed = LiveFeed(db)
    live_feed.update_employee_entry_today_by_db()
    live_feed.update_id_to_name_dict_by_db()

    faces_detected_dir='\\'.join([os.getcwd(),'faces detected in live feed'])
    if not os.path.isdir(faces_detected_dir):
        os.mkdir(faces_detected_dir)

    no_faces_detected_dir='\\'.join([os.getcwd(),'no faces detected in live feed'])
    if not os.path.isdir(no_faces_detected_dir):
        os.mkdir(no_faces_detected_dir)


    cap = cv.VideoCapture(0,cv.CAP_DSHOW)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        if (frame is  None) or (isinstance(frame, type(None))) :
            continue
        frame_image = CapturedFrame(cv.resize(frame, None, fx=0.5, fy=0.5,interpolation=cv.INTER_AREA))
        frame_image.set_face_image()
        if frame_image.face_detected:
            print("face detected")
            frame_image.face_image.save(os.path.join(faces_detected_dir,str(CapturedFrame.number_of_faces_detected)+".jpg"))
            frame_image.identify()
            if frame_image.face_recognized:
                print("".join(["face recognized as employee id=",str(frame_image.id_detected)]))
                if not live_feed.employees_entry_today[frame_image.id_detected-1]:
                    now = datetime.now().strftime('%Y %m %d %H %M %S').split(' ')
                    live_feed.register_entry(frame_image.id_detected)
                    live_feed.employees_entry_today[frame_image.id_detected-1]=True
                else:
                    print("".join(["employee id=", str(frame_image.id_detected)," already registered entry today"]))
            else:
                print("no face recognized")
        else:
            print("no face detected")
            frame_image.save("".join([no_faces_detected_dir,"\\",str(CapturedFrame.number_of_face_not_detected),".jpg"]))
        c = cv.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv.destroyAllWindows()