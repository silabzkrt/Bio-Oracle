import numpy as np

class CellTracker:
    def __init__(self, movement_threshold=50, staying_frame_count=30, max_history=100):
        self.tracks = {}
        self.next_id = 0
        self.move_thresh = movement_threshold
        self.stay_thresh = staying_frame_count

    def update(self, detections):
        updated_tracks = []
        for det in detections:
            bbox = det['bbox']
            center = (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2))
            
            match_id = None
            for tid, data in self.tracks.items():
                dist = np.linalg.norm(np.array(center) - np.array(data['history'][-1]))
                if dist < 50: 
                    match_id = tid
                    break
            
            if match_id is None:
                match_id = self.next_id
                self.next_id += 1
                self.tracks[match_id] = {'history': [center], 'frames': 0}
            else:
                self.tracks[match_id]['history'].append(center)
                self.tracks[match_id]['frames'] += 1

            first_pos = self.tracks[match_id]['history'][0]
            total_dist = np.linalg.norm(np.array(center) - np.array(first_pos))
            status = 'moving' if total_dist > self.move_thresh else 'staying'
            
            updated_tracks.append({
                'track_id': match_id,
                'bbox': bbox,
                'status': status
            })
        return updated_tracks

    def get_counts(self):
        return self.tracks