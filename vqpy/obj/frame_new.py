class Frame:
    def __init__(self, video_meta_data):
        self.video_meta_data = video_meta_data

    # video reader
    def update_id_image(self, id, image):
        self.frame_id = id
        self.frame_image = image
