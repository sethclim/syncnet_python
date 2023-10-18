#!/usr/bin/python

import sys, time, os, pdb, argparse, pickle, subprocess, glob, cv2
import numpy as np
from shutil import rmtree

import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect import open_video
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal

from .detectors import S3FD

class RunPipeline():

  def __init__(self, opt):
    self.opt = opt
    self.DET = S3FD(device='cuda')

  # ========== ========== ========== ==========
  # # IOU FUNCTION
  # ========== ========== ========== ==========
  def bb_intersection_over_union(self, boxA, boxB):
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
  
    interArea = max(0, xB - xA) * max(0, yB - yA)
  
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
  
    iou = interArea / float(boxAArea + boxBArea - interArea)
  
    return iou

  # ========== ========== ========== ==========
  # # FACE TRACKING
  # ========== ========== ========== ==========
  def track_shot(self, opt,scenefaces):

    iouThres  = 0.5     # Minimum IOU between consecutive face detections
    tracks    = []

    count = 0
    while True:
      count = count + 1

      track     = []
      for framefaces in scenefaces:
  
        for face in framefaces:

          if track == []:
      
            track.append(face)
            framefaces.remove(face)
          elif face['frame'] - track[-1]['frame'] <= opt.num_failed_det:

            iou = self.bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
            if iou > iouThres:
        
              track.append(face)
      
              framefaces.remove(face)
      
              continue
          else:
            break


      if track == []:
        break
      elif len(track) > opt.min_track:
        framenum    = np.array([ f['frame'] for f in track ])
        bboxes      = np.array([np.array(f['bbox']) for f in track])

        frame_i   = np.arange(framenum[0],framenum[-1]+1)

        bboxes_i    = []
        for ij in range(0,4):
          interpfn  = interp1d(framenum, bboxes[:,ij])
          bboxes_i.append(interpfn(frame_i))
        bboxes_i  = np.stack(bboxes_i, axis=1)

        face_size = np.mean(bboxes_i[:,2]-bboxes_i[:,0]), np.mean(bboxes_i[:,3]-bboxes_i[:,1])
        print(face_size)
        if max(face_size) > opt.min_face_size:
          print("Passed Face Size")
          tracks.append({'frame':frame_i,'bbox':bboxes_i})

    print("TRACKS ", len(tracks))
    return tracks

  # ========== ========== ========== ==========
  # # VIDEO CROP AND SAVE
  # ========== ========== ========== ==========
  def crop_video(self, opt,track,cropfile):
    print("CROP VIDEO")
    flist = glob.glob(os.path.join(opt.frames_dir,opt.reference,'*.jpg'))
    flist.sort()

    # print(flist)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vOut = cv2.VideoWriter(cropfile+'t.avi', fourcc, opt.frame_rate, (224,224))

    dets = {'x':[], 'y':[], 's':[]}

    for det in track['bbox']:

      dets['s'].append(max((det[3]-det[1]),(det[2]-det[0]))/2) 
      dets['y'].append((det[1]+det[3])/2) # crop center x 
      dets['x'].append((det[0]+det[2])/2) # crop center y

    # Smooth detections
    dets['s'] = signal.medfilt(dets['s'],kernel_size=13)   
    dets['x'] = signal.medfilt(dets['x'],kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'],kernel_size=13)

    for fidx, frame in enumerate(track['frame']):

      cs  = opt.crop_scale

      bs  = dets['s'][fidx]   # Detection box size
      bsi = int(bs*(1+2*cs))  # Pad videos by this amount 

      image = cv2.imread(flist[frame])
      
      frame = np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))
      my  = dets['y'][fidx]+bsi  # BBox center Y
      mx  = dets['x'][fidx]+bsi  # BBox center X

      face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
      
      vOut.write(cv2.resize(face,(224,224)))

    audiotmp    = os.path.join(opt.tmp_dir,opt.reference,'audio.wav')
    audiostart  = (track['frame'][0])/opt.frame_rate
    audioend    = (track['frame'][-1]+1)/opt.frame_rate

    vOut.release()

    # ========== CROP AUDIO FILE ==========
    command = ["ffmpeg", "-y", 
              "-i",os.path.join(opt.avi_dir,opt.reference,'audio.wav'), 
              "-ss", "{:.3f}".format(audiostart), 
              "-to","{:.3f}".format(audioend), 
              audiotmp]
    
    output = subprocess.run(args=command, capture_output=True, shell=False)

    if output.returncode != 0:
      print(output.returncode)
      print(output.stderr)
      pdb.set_trace()

    sample_rate, audio = wavfile.read(audiotmp)

    # ========== COMBINE AUDIO AND VIDEO FILES ==========
    command = ["ffmpeg", "-y", "-i", f"{cropfile}t.avi", "-i", audiotmp, "-c:v", "copy", "-c:a", "copy", f"{cropfile}.avi"]
    output = subprocess.run(command, capture_output=True)

    if output.returncode != 0:
      print(output.returncode)
      print(output.stderr)
      pdb.set_trace()

    print('Written %s'%cropfile)

    os.remove(cropfile+'t.avi')

    print('Mean pos: x %.2f y %.2f s %.2f'%(np.mean(dets['x']),np.mean(dets['y']),np.mean(dets['s'])))

    return {'track':track, 'proc_track':dets}

  # ========== ========== ========== ==========
  # # FACE DETECTION
  # ========== ========== ========== ==========
  def inference_video(self, opt):
    flist = glob.glob(os.path.join(opt.frames_dir,opt.reference,'*.jpg'))
    flist.sort()

    dets = []
        
    for fidx, fname in enumerate(flist):

      start_time = time.time()
      
      image = cv2.imread(fname)

      image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      bboxes = self.DET.detect_faces(image_np, conf_th=0.9, scales=[opt.facedet_scale])

      dets.append([]);
      for bbox in bboxes:
        dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})

      elapsed_time = time.time() - start_time

      # print('%s-%05d; %d dets; %.2f Hz' % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),fidx,len(dets[-1]),(1/elapsed_time))) 

    savepath = os.path.join(opt.work_dir,opt.reference,'faces.pckl')

    with open(savepath, 'wb') as fil:
      pickle.dump(dets, fil)

    return dets

  # ========== ========== ========== ==========
  # # SCENE DETECTION
  # ========== ========== ========== ==========
  def scene_detect(self, opt):

    video_stream = open_video(os.path.join(opt.avi_dir,opt.reference,'video.avi'))
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    # Add ContentDetector algorithm (constructor takes detector options like threshold).
    scene_manager.add_detector(ContentDetector())

    scene_manager.auto_downscale = True
    #video_manager.set_downscale_factor()

    # video_manager.start()

    scene_manager.detect_scenes(video=video_stream)

    scene_list = scene_manager.get_scene_list(start_in_scene=True)

    savepath = os.path.join(opt.work_dir, opt.reference, 'scene.pckl')

    # if scene_list == []:
    #   scene_list = [(video_stream.base_timecode,video_stream.frame_rate)]

    with open(savepath, 'wb') as fil:
      pickle.dump(scene_list, fil)

    print('%s - scenes detected %d'%(os.path.join(opt.avi_dir,opt.reference,'video.avi'),len(scene_list)))

    return scene_list
      
  def process(self, data_dir, reference, min_track, video_file):
    self.opt.data_dir  = data_dir
    self.opt.reference = reference
    self.opt.min_track = min_track
    self.opt.videofile = video_file

    # ========== DELETE EXISTING DIRECTORIES ==========
    if os.path.exists(os.path.join(self.opt.work_dir,self.opt.reference)):
      rmtree(os.path.join(self.opt.work_dir,self.opt.reference))

    if os.path.exists(os.path.join(self.opt.crop_dir,self.opt.reference)):
      rmtree(os.path.join(self.opt.crop_dir,self.opt.reference))

    if os.path.exists(os.path.join(self.opt.avi_dir,self.opt.reference)):
      rmtree(os.path.join(self.opt.avi_dir,self.opt.reference))

    if os.path.exists(os.path.join(self.opt.frames_dir,self.opt.reference)):
      rmtree(os.path.join(self.opt.frames_dir,self.opt.reference))

    if os.path.exists(os.path.join(self.opt.tmp_dir,self.opt.reference)):
      rmtree(os.path.join(self.opt.tmp_dir,self.opt.reference))

    # ========== MAKE NEW DIRECTORIES ==========
    os.makedirs(os.path.join(self.opt.work_dir,self.opt.reference))
    os.makedirs(os.path.join(self.opt.crop_dir,self.opt.reference))
    os.makedirs(os.path.join(self.opt.avi_dir,self.opt.reference))
    os.makedirs(os.path.join(self.opt.frames_dir,self.opt.reference))
    os.makedirs(os.path.join(self.opt.tmp_dir,self.opt.reference))

    # ========== CONVERT VIDEO AND EXTRACT FRAMES ==========

    command = ["ffmpeg", "-y", 
               "-i", self.opt.videofile, 
               "-qscale:v", "2", 
               "-async", "1",
               "-r", "25", 
               os.path.join(self.opt.avi_dir, self.opt.reference,"video.avi")]
    
    output = subprocess.run(args=command)

    command = ["ffmpeg", "-y", "-i", os.path.join(self.opt.avi_dir, self.opt.reference,'video.avi'), "-qscale:v", "2", "-threads", "1", "-f", "image2", os.path.join(self.opt.frames_dir,self.opt.reference,'%06d.jpg')] 
    output = subprocess.run(args=command)

    command = ["ffmpeg", "-y", "-i", os.path.join(self.opt.avi_dir,self.opt.reference,'video.avi'), "-ac", "1", "-vn", "-acodec", "pcm_s16le", "-ar", "16000", os.path.join(self.opt.avi_dir,self.opt.reference,'audio.wav')]
    output = subprocess.run(args=command)

    # ========== FACE DETECTION ==========
    faces = self.inference_video(self.opt)

    # ========== SCENE DETECTION ==========
    scene = self.scene_detect(self.opt)

    # ========== FACE TRACKING ========== 
    alltracks = []
    vidtracks = []

    for shot in scene:
      # print("Shot 1 ", shot[1].frame_num)
      # print("Shot 2 ", shot[0].frame_num)
      if shot[1].frame_num - shot[0].frame_num >= self.opt.min_track :
        # print("Passed")
        alltracks.extend(self.track_shot(self.opt,faces[shot[0].frame_num:shot[1].frame_num]))

    # ========== FACE TRACK CROP ==========
    for ii, track in enumerate(alltracks):
      vidtracks.append(self.crop_video(self.opt,track,os.path.join(self.opt.crop_dir, self.opt.reference,'%05d'%ii)))



    # ========== SAVE RESULTS ==========
    savepath = os.path.join(self.opt.work_dir, self.opt.reference,'tracks.pckl')

    with open(savepath, 'wb') as fil:
      pickle.dump(vidtracks, fil)

    rmtree(os.path.join(self.opt.tmp_dir, self.opt.reference))
