from collections import deque
import concurrent.futures
from threading import Lock
import os
from PIL import Image
from timing_utils import time_it

class ImageSaver:
    def __init__(self, max_queue_size=1000):
        self.image_save_queue = deque()
        self.max_queue_size = max_queue_size
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.lock = Lock()
        # Keep track of saved images and pending saves
        self.saved_images = set()
        self.pending_images = set()
        
    def _save_image_worker(self, image, path):
        """Worker function to save image in a separate thread"""
        try:
            image.save(path)
            with self.lock:
                self.saved_images.add(path)
                self.pending_images.remove(path)
            return True
        except Exception as e:
            print(f"Error saving image to {path}: {e}")
            with self.lock:
                self.pending_images.remove(path)
            return False
    
    def queue_image_save(self, image, path):
        """Add image to save queue"""
        with self.lock:
            self.image_save_queue.append((image, path))
            self.pending_images.add(path)
    
    def is_image_ready(self, path):
        """Check if image is available for reading"""
        return path in self.saved_images
    
    @time_it
    def process_save_queue(self):
        """Process pending image saves in batch"""
        if not self.image_save_queue:
            return
            
        with self.lock:
            # Process up to 100 images at once
            batch_size = min(100, len(self.image_save_queue))
            items_to_process = [self.image_save_queue.popleft() for _ in range(batch_size)]
        
        # Submit all saves to thread pool
        futures = [
            self.executor.submit(self._save_image_worker, image, path)
            for image, path in items_to_process
        ]
        
        # Wait for batch to complete
        concurrent.futures.wait(futures)