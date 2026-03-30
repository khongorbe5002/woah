import cv2
import time
import numpy as np
import subprocess
import multiprocessing as mp
from ultralytics import YOLO
from evdev import InputDevice, categorize, ecodes
import threading

DEVICE_PATH = "/dev/input/event5"

def speak_text(text):
    try:
        subprocess.call(["/usr/bin/espeak","-s","200","-v","en+f3",text])
    except Exception as e:
        print(f"Speech error: {e}")

def headphone_listener():
    dev = InputDevice(DEVICE_PATH)
    global current_mode

    print("Headphone controls active")

    for event in dev.read_loop():
        if event.type == ecodes.EV_KEY:
            key_event = categorize(event)

            if key_event.keystate == 1:
                key = key_event.keycode

                if key == "KEY_VOLUMEUP":
                    current_mode = (current_mode % NUM_MODES) + 1
                    speak_text(MODES[current_mode]["name"])
                    print("Mode:", MODES[current_mode]["name"])

                elif key == "KEY_VOLUMEDOWN":
                    current_mode = (current_mode - 2) % NUM_MODES + 1
                    speak_text(MODES[current_mode]["name"])
                    print("Mode:", MODES[current_mode]["name"])

                elif key in ["KEY_PLAYCD", "KEY_PAUSECD"]:
                    print("Scene triggered")
                    describe_scene([])

def get_direction_descriptor(cx,cy,frame_w,frame_h):
    third_w=frame_w/3
    third_h=frame_h/3
    horiz="left" if cx<third_w else "center" if cx<2*third_w else "right"
    vert="upper" if cy<third_h else "center" if cy<2*third_h else "bottom"
    if vert=="center" and horiz=="center": return "center"
    if vert=="center": return horiz
    if horiz=="center": return vert
    return f"{vert} {horiz}"

OBSTACLE_CLASSES=["Bike","Car","Chair","Emergency Blue Phone","Exit sign","Person","Pole","Stairs","Tree","Washroom"]

MODES={
    1:{"name":"Normal","excluded":{"Person","Exit sign"},"catch_unknown":False},
    2:{"name":"Everything","excluded":set(),"catch_unknown":True},
    3:{"name":"Emergency","excluded":set(),"catch_unknown":False},
    4:{"name":"Scene Mode","excluded":set(),"catch_unknown":False},
}

current_mode=1
NUM_MODES=len(MODES)

def get_active_classes():
    return set(OBSTACLE_CLASSES)-MODES[current_mode]["excluded"]

def mode_catches_unknown():
    return MODES[current_mode]["catch_unknown"]

tracked_objects=[]
TRACKED_OBJECT_MOVE_THRESHOLD_PX=80
TRACKED_OBJECT_MAX_AGE=5.0
ALERT_REMINDER_SECONDS=1.0

last_speech_time=0
SPEECH_COOLDOWN=1.0

def _euclidean_dist(p0,p1):
    return ((p0[0]-p1[0])**2+(p0[1]-p1[1])**2)**0.5

def _find_tracked_object(label,center):
    for obj in tracked_objects:
        if obj["label"]==label and _euclidean_dist(center,obj["center"])<TRACKED_OBJECT_MOVE_THRESHOLD_PX:
            return obj
    return None

def _cleanup_tracked_objects(now):
    global tracked_objects
    tracked_objects=[o for o in tracked_objects if now-o["last_seen"]<TRACKED_OBJECT_MAX_AGE]

def safe_speak(text):
    global last_speech_time
    now=time.time()
    if now-last_speech_time>SPEECH_COOLDOWN:
        speak_text(text)
        last_speech_time=now

def describe_scene(detections):
    if len(detections)==0:
        safe_speak("No major objects detected")
        return
    counts={}
    for d in detections:
        label=d["label"]
        counts[label]=counts.get(label,0)+1
    parts=[]
    for k,v in counts.items():
        parts.append(f"a {k}" if v==1 else f"{v} {k}s")
    safe_speak("I see "+", ".join(parts))

def draw_sensor(sensor_data):
    size=400
    cell=size//8
    img=np.zeros((size,size,3),dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            d=sensor_data[r,c]
            if d==0:
                color=(50,50,50)
            else:
                v=min(d,3000)/3000
                color=(int(255*v),int(255*(1-v)),0)
            x1,y1=c*cell,r*cell
            x2,y2=x1+cell,y1+cell
            cv2.rectangle(img,(x1,y1),(x2,y2),color,-1)
    return img

def run_sensor_process(shared_array,lock):
    from working_cam_sensor import VL53L5CXSensor
    try:
        sensor=VL53L5CXSensor(verbose=False)
    except Exception as e:
        print(f"Sensor failed: {e}")
        return
    while True:
        data=sensor.get_ranging_data()
        if data is not None:
            flat=data.flatten()
            with lock:
                for i in range(64):
                    shared_array[i]=flat[i]
        time.sleep(0.033)

if __name__=="__main__":
    shared_sensor_data=mp.Array('i',64)
    sensor_lock=mp.Lock()

    p=mp.Process(target=run_sensor_process,args=(shared_sensor_data,sensor_lock))
    p.daemon=True
    p.start()

    threading.Thread(target=headphone_listener, daemon=True).start()

    model=YOLO("best.pt")

    cap=cv2.VideoCapture(0,cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Camera failed")
        exit()

    print("System running (headphone controls enabled)")

    try:
        while True:
            ret,frame=cap.read()
            if not ret:
                break

            with sensor_lock:
                sensor_data=np.array(shared_sensor_data[:],dtype=np.int32).reshape((8,8))

            results=model(frame,verbose=False)

            active_classes=get_active_classes()
            catch_unknown=mode_catches_unknown()

            frame_now=time.time()
            _cleanup_tracked_objects(frame_now)

            covered_cells=set()
            current_detections=[]

            for r in results:
                for b in r.boxes:
                    x1,y1,x2,y2=map(int,b.xyxy[0])
                    cls=int(b.cls[0])
                    label=model.names[cls]

                    cx=int((x1+x2)/2)
                    cy=int((y1+y2)/2)

                    gx=int(cx/frame.shape[1]*8)
                    gy=int(cy/frame.shape[0]*8)
                    gx=max(0,min(7,gx))
                    gy=max(0,min(7,gy))

                    dist=sensor_data[gy,gx]
                    covered_cells.add((gy,gx))

                    if label in active_classes:
                        display_label=label
                    elif catch_unknown:
                        display_label="object"
                    else:
                        continue

                    current_detections.append({"label":display_label})

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                    if 0<dist<1000 and current_mode!=4:
                        obj=_find_tracked_object(display_label,(cx,cy))
                        if obj is None:
                            tracked_objects.append({"label":display_label,"center":(cx,cy),
                                                    "last_alert":frame_now,"last_seen":frame_now})
                            direction=get_direction_descriptor(cx,cy,frame.shape[1],frame.shape[0])
                            safe_speak(f"{display_label} {direction}")

            if current_mode==4:
                describe_scene(current_detections)

            cv2.imshow("Camera",frame)
            cv2.imshow("Sensor",draw_sensor(sensor_data))

            if cv2.waitKey(1)==27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        p.terminate()
        p.join()
