import logging
from collections import deque
import concurrent.futures
from threading import Lock
from PIL import Image
import os
import hashlib
import io
import time
import sys  # Added for debugging

from timing_utils import time_it

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ImageSaver:
    """
    Asynchronous image saving using a thread pool executor.
    Helps reduce blocking on disk I/O.
    """
    def __init__(self, log_dir=None, max_queue_size=1000):
        """
        Parameters
        ----------
        log_dir : str, optional
            The base directory for logs, used to create the 'images' subdirectory.
            If None, images must be saved with absolute paths.
        max_queue_size : int
            Maximum number of images to keep in the queue before forced processing
        """
        print("Initializing ImageSaver...")
        self.image_save_queue = deque()
        self.max_queue_size = max_queue_size
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.lock = Lock()
        self.saved_images = set()
        self.pending_images = set()
        self.saved_hashes = set()  # Track image hashes to avoid duplicates
        
        # Handle directory setup - simplified approach
        self.image_dir = None
        if log_dir:
            self.image_dir = os.path.join(log_dir, "images")
            os.makedirs(self.image_dir, exist_ok=True)
            print(f"Created main image directory: {self.image_dir}")
            
            # Only create the domain directory upfront
            domain_dir = os.path.join(self.image_dir, "domain")
            os.makedirs(domain_dir, exist_ok=True)
            print(f"Created domain directory: {domain_dir}")
        
        print("ImageSaver initialization complete")

    def _generate_hash(self, image):
        """
        Generate SHA256 hash of image content for de-duplication.
        """
        try:
            # Convert to a consistent format first
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            return hashlib.sha256(img_byte_arr).hexdigest()
        except Exception as e:
            logger.error(f"Error generating image hash: {e}")
            return None

    def _save_image_worker(self, image, path):
        """
        Worker function run in a separate thread to save an image to disk.
        """
        try:
            # Ensure the directory exists
            dir_path = os.path.dirname(path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            
            # Convert to RGB if needed (prevents some PIL errors)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            image.save(path)
            with self.lock:
                self.saved_images.add(path)
                if path in self.pending_images:
                    self.pending_images.remove(path)
            return True
        except Exception as e:
            logger.error(f"Error saving image to {path}: {e}")
            with self.lock:
                if path in self.pending_images:
                    self.pending_images.remove(path)
            return False

    def queue_image_save(self, image, path):
        """
        Queue an image for asynchronous saving.

        Parameters
        ----------
        image : PIL.Image
        path : str
        """
        with self.lock:
            self.image_save_queue.append((image, path))
            self.pending_images.add(path)

    def queue_agent_image_save(self, image, agent_id, metadata=None):
        """
        Save an image to an agent's directory with simple filename
        """
        try:
            if self.image_dir is None:
                return None
                
            if image is None:
                return None
                
            # Create agent directory if it doesn't exist
            agent_dir = os.path.join(self.image_dir, f"agent_{agent_id}")
            if not os.path.exists(agent_dir):
                os.makedirs(agent_dir, exist_ok=True)
            
            # Use simple timestamp-based filename to avoid path issues
            timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
            filename = f"img_{timestamp}.png"
            
            # Construct relative and full paths
            rel_path = os.path.join("images", f"agent_{agent_id}", filename)
            full_path = os.path.join(self.image_dir, f"agent_{agent_id}", filename)
            
            # Queue for saving
            self.queue_image_save(image, full_path)
            
            return rel_path
        except Exception as e:
            print(f"ERROR in queue_agent_image_save: {e}", file=sys.stderr)
            return None

    def queue_domain_image_save(self, image, metadata=None):
        """
        Save an image to the domain directory with simple filename
        """
        try:
            if self.image_dir is None:
                return None
                
            if image is None:
                return None
            
            # Use simple timestamp-based filename to avoid path issues
            timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
            filename = f"domain_{timestamp}.png"
            
            # Construct relative and full paths
            rel_path = os.path.join("images", "domain", filename)
            full_path = os.path.join(self.image_dir, "domain", filename)
            
            # Queue for saving
            self.queue_image_save(image, full_path)
            
            return rel_path
        except Exception as e:
            print(f"ERROR in queue_domain_image_save: {e}", file=sys.stderr)
            return None

    def queue_image_save_by_hash(self, image):
        """
        Backward compatibility method - choose where to save based on metadata
        """
        return self.queue_domain_image_save(image)

    def is_image_ready(self, path):
        """
        Check if the image at the given path is fully saved to disk.
        """
        return path in self.saved_images or os.path.exists(path)

    @time_it
    def process_save_queue(self):
        """
        Process a batch of pending image saves using the thread pool.
        """
        with self.lock:
            if not self.image_save_queue:
                return
            batch_size = min(100, len(self.image_save_queue))
            items_to_process = [self.image_save_queue.popleft() for _ in range(batch_size)]

        futures = [
            self.executor.submit(self._save_image_worker, image, path)
            for image, path in items_to_process
        ]
        concurrent.futures.wait(futures)
        
    def stop(self):
        """
        Process all remaining images in the queue and shut down the executor.
        Call this before program exit.
        """
        logger.info("Processing remaining image save queue...")
        
        # Process all remaining images
        while self.image_save_queue:
            self.process_save_queue()
            
        # Shutdown executor
        self.executor.shutdown(wait=True)
        logger.info("ImageSaver stopped.")