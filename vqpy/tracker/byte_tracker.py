from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from vqpy.video_loader import FrameStream

from . import matching
from .base_track import BaseTrack, TrackState
from .kalman_filter import KalmanFilter

class ByteTracker(object):
    shared_kalman = KalmanFilter()
    
    input_fields = ["tlbr", "score"]
    output_fields = ["track_id"]
    
    class Data(BaseTrack):
        def __init__(self, data: Dict): 
            """Create an instance of ByteTracker Data field"""
            self.track_id = self.next_id()
            # It is possible to remove duplicated (track_id)s for the same tracks
            self.data = data
            self._tlbr = np.asarray(data["tlbr"], dtype=np.float)
            self.score = data["score"]
            
            self.is_activated = False
            self.tracklet_len = 0
        
        def set_tlbr(self, tlbr):
            self._tlbr = np.asarray(tlbr, dtype=np.float)
        
        def initiate(self, frame_id):
            """Initiate a VObj, so that it have tracking property

            Args:
                frame: the image of the frame
                frame_id (int): the ID of frame
            """
            self.kalman_filter = None
            self.mean = None
            self.covariance = None
            
            self.tracklet_len = 0
            self.state = TrackState.Tracked
            if frame_id == 1:
                self.is_activated = True
            self.frame_id = frame_id
            self.start_frame = frame_id
        
        def update(self, frame_id, new_track: ByteTracker.Data, reactivate=False, newid=False):
            """Update a vobject, executed for all activated object in one frame"""
            self.data = new_track.data
            self._tlbr = new_track._tlbr
            self.score = new_track.score
            if reactivate:
                self.tracklet_len = 0
            else:
                self.tracklet_len += 1
            self.state = TrackState.Tracked
            self.is_activated = True
            self.frame_id = frame_id
            if newid:
                self.track_id = self.next_id()
        
        @property
        # @jit(nopython=True)
        def tlbr(self):
            """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
            `(top left, bottom right)`.
            """
            return self._tlbr.copy()

        @staticmethod
        # @jit(nopython=True)
        def tlbr_to_tlwh(tlbr):
            """Convert bounding box to format `(top left x, top left y, width, height)`.
            """
            ret = np.asarray(tlbr).copy()
            ret[2:] -= ret[:2]
            return ret

        @staticmethod
        # @jit(nopython=True)
        def tlbr_to_xyah(tlbr):
            """Convert bounding box to format `(center x, center y, aspect ratio,
            height)`, where the aspect ratio is `width / height`.
            """
            ret = np.asarray(tlbr).copy()
            ret[2:] -= ret[:2]
            ret[:2] += ret[2:] / 2
            ret[2] /= ret[3]
            return ret
        
        @staticmethod
        # @jit(nopython=True)
        def xyah_to_tlbr(xyah):
            ret = np.asarray(xyah).copy()
            ret[2] *= ret[3]
            ret[:2] -= ret[2:] / 2
            ret[2:] += ret[:2]
            return ret
        
        @property
        # @jit(nopython=True)
        def tlwh(self):
            return ByteTracker.Data.tlbr_to_tlwh(self.tlbr)

        @property
        # @jit(nopython=True)
        def xyah(self):
            return ByteTracker.Data.tlbr_to_xyah(self.tlbr)
        
        def extract_data(self) -> Dict:
            ret = self.data.copy()
            for _field in ByteTracker.output_fields:
                ret[_field] = getattr(self, _field)
            return ret
    
    def __init__(self, ctx: FrameStream):
        self.ctx = ctx

        self.track_thresh = 0.6
        self.det_thresh = self.track_thresh + 0.1
        self.match_thresh = 0.9
        self.buffer_size = int(ctx.fps / 30.0 * 30)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        
        self.tracked_stracks: List[ByteTracker.Data] = []
        self.lost_stracks: List[ByteTracker.Data] = []
        self.removed_stracks: List[ByteTracker.Data] = []
    
    def _predict(self, track : Data):
        mean_state = track.mean.copy()
        if track.state != TrackState.Tracked:
            mean_state[7] = 0
        track.mean, track.covariance = track.kalman_filter.predict(mean_state, track.covariance)
    
    def _multipredict(self, tracks : List[Data]):
        if len(tracks) > 0:
            multi_mean = np.asarray([track.mean.copy() for track in tracks])
            multi_covariance = np.asarray([track.covariance for track in tracks])
            for i, track in enumerate(tracks):
                if track.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = ByteTracker.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                tracks[i].mean = mean
                tracks[i].covariance = cov
    
    def _initiate(self, track : Data, kalman_filter, frame_id):
        track.initiate(frame_id)
        track.kalman_filter = kalman_filter
        track.mean, track.covariance = track.kalman_filter.initiate(track.xyah)
    
    def _update(self, frame_id, track : Data, new_track : Data, reactivate=False, newid=False):
        track.update(frame_id, new_track, reactivate, newid)
        track.mean, track.covariance = track.kalman_filter.update(track.mean, track.covariance, track.xyah)
        track.set_tlbr(ByteTracker.Data.xyah_to_tlbr(track.mean[:4]))
    
    def update(self, datas: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        frame_id = self.ctx.frame_id
        dets: List[ByteTracker.Data] = [ByteTracker.Data(x) for x in datas]
        
        activated_stracks: List[ByteTracker.Data] = []
        refind_stracks: List[ByteTracker.Data] = []
        lost_stracks: List[ByteTracker.Data] = []
        removed_stracks: List[ByteTracker.Data] = []
        
        dets_high = [x for x in dets if x.score > self.track_thresh]
        dets_low = [x for x in dets if self.track_thresh >= x.score > 0.1]
        
        '''Step 1: Add newly detected tracklets to tracked_stracks'''
        unconfirmed: List[ByteTracker.Data] = []
        tracked_stracks: List[ByteTracker.Data] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        self._multipredict(strack_pool)
        dists = matching.iou_distance(strack_pool, dets_high)
        dists = matching.fuse_score(dists, dets_high)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            new_track = dets_high[idet]
            if track.state == TrackState.Tracked:
                self._update(frame_id, track, new_track)
                activated_stracks.append(track)
            else:
                self._update(frame_id, track, new_track, reactivate=True)
                refind_stracks.append(track)
        
        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, dets_low)
        matches, u_track, u_detection_low = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            new_track = dets_low[idet]
            if track.state == TrackState.Tracked:
                self._update(frame_id, track, new_track)
                activated_stracks.append(track)
            else:
                self._update(frame_id, track, new_track, reactivate=True)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        dets_rem = [dets_high[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, dets_rem)
        dists = matching.fuse_score(dists, dets_rem)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            new_track = dets_rem[idet]
            self._update(frame_id, unconfirmed[itracked], new_track)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = dets_rem[inew]
            if track.score < self.det_thresh:
                continue
            self._initiate(track, self.kalman_filter, frame_id)
            activated_stracks.append(track)
        
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        
        return [x.extract_data() for x in self.tracked_stracks], [x.extract_data() for x in self.lost_stracks]

def joint_stracks(tlista : List[ByteTracker.Data], tlistb : List[ByteTracker.Data]) -> List[ByteTracker.Data]:
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista : List[ByteTracker.Data], tlistb : List[ByteTracker.Data]) -> List[ByteTracker.Data]:
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa : List[ByteTracker.Data], stracksb : List[ByteTracker.Data]) -> Tuple[List[ByteTracker.Data], List[ByteTracker.Data]]:
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
