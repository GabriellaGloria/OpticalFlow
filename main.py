#!/usr/bin/env python

"""
Lucas-Kanade tracker
====================
Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.
Usage
-----
lk_track.py [<video_source>]
Keys
----
ESC - exit
"""

import sys
import math
import numpy as np
import cv2 as cv

import video
from common import anorm2, draw_str

# Python 2/3 compatibility
#from __future__ import print_function

class App:
    """
    The main application class that runs the Lucas-Kanade tracker
    on the provided video source.
    """
    def __init__(self, video_src):
        self.track_len = 2
        self.detect_interval = 4
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.alpha = 0.5
        self.frame_idx = 0

    def run(self):
        cnt = 0
        """
        This method contains the main loop that processes each frame
        and applies the Lucas-Kanade tracking algorithm.
        """
        # Lucas-Kanade parameters
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        feature_params = dict(maxCorners=500,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)

        # Constants
        fps = 30
        px2m1 = 0.1
        ms2kmh = 3.6

        ret, first_frame = self.cam.read()
        cal_mask = np.zeros_like(first_frame[:, :, 0])
        view_mask = np.zeros_like(first_frame[:, :, 0])
        cal_polygon = np.array([[0, 0], [0, 2500], [1080, 2500], [1080, 0]])
        view_polygon = np.array([[0, 0], [0, 2500], [1080, 2500], [1080, 0]])
        prv1 = 0
        prn1 = 0
        ptn1 = 0

        polygon1 = np.array([[0, 0], [0, 2500], [1080, 2500], [1080, 0]])
        cv.fillConvexPoly(cal_mask, cal_polygon, 1)
        cv.fillConvexPoly(view_mask, view_polygon, 1)

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter("output.mp4", fourcc, 30.0, (1080, 1920))

        while (self.cam.isOpened()):
            _ret, frame = self.cam.read()
            if _ret:
                # both video from webcam and input video will go here and run 
                print("runned")
                vis = frame.copy()
                cmask = frame.copy()

                mm1 = 0
                v1 = 0

                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame_gray = cv.bitwise_and(frame_gray, frame_gray, mask=cal_mask)

                vis = cv.bitwise_and(vis, vis, mask=view_mask)

                draw_str(vis, (30, 40), 'speed %d cm/s' % prv1)

                draw_str(vis, (900, 40), 'ptn1: %d' % prn1)

                if len(self.tracks) > 0:
                    img0, img1 = self.prev_gray, frame_gray
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                    p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                    d = abs(p0-p0r).reshape(-1, 2).max(-1)
                    good = d < 1
                    new_tracks = []
                    for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                        if not good_flag:
                            continue
                        tr.append((x, y))
                        if len(tr) > self.track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        cv.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
                    self.tracks = new_tracks

                    ptn1, ptn2, ptn3, ptn4, ptn5 = 0, 0, 0, 0, 0
                    import time
                    start = time.time()  
                    for idx, tr in enumerate(self.tracks):
                        result_polygon1 = cv.pointPolygonTest(polygon1, tr[0],True)
                        if result_polygon1 > 0:
                            ptn1 += 1
                            dif1 = tuple(map(lambda i, j: i - j, tr[0], tr[1]))
                            mm1 += math.sqrt(dif1[0]*dif1[0] + dif1[1]*dif1[1])
                            mmm1 = mm1/ptn1
                            v1 = mmm1*px2m1*fps*ms2kmh*100000/3600/100
                    cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 0, 255))

                prn1 = ptn1

                if self.frame_idx % self.detect_interval == 0:
                    if ptn1 > 10:
                        draw_str(vis, (900, 40), 'ptn1: %d' % ptn1)
                        draw_str(vis, (30, 40), 'speed %d cm/s' % v1)

                    # Speed writing part
                    prv1 = v1
                    mask = np.zeros_like(frame_gray)
                    mask[:] = 255
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                        cv.circle(mask, (int(x), int(y)), 4, 0, -1)
                    p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])


                self.frame_idx += 1
                self.prev_gray = frame_gray
                cv.addWeighted(cmask, self.alpha, vis, 1 - self.alpha, 0, vis)
                cnt += 1
                print(cnt)
                out.write(vis)
                cv.imshow('Optical Flow - Lucas Kanade', vis)
                #cv.waitKey(0)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        out.release()
        self.cam.release()
        cv.destroyAllWindows()

def main():
    """
    The main entry point of the application.
    """
    try:
        video_src = sys.argv[1]
    except IndexError:
        video_src = 0

    app = App(video_src)
    app.run()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()