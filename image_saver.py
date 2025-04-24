import logging
from collections import deque
import concurrent.futures
from threading import Lock
from PIL import Image

from timing_utils import time_it

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ImageSaver:
    """
    Asynchronous image saving using a thread pool executor.
    Helps reduce blocking on disk I/O.
    """
    def __init__(self, max_queue_size=1000):
        """
        Parameters
        ----------
        max_queue_size : int
            Maximum number of images to keep in the queue before forced processing
        """
        self.image_save_queue = deque()
        self.max_queue_size = max_queue_size
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.lock = Lock()
        self.saved_images = set()
        self.pending_images = set()

    def _save_image_worker(self, image, path):
        """
        Worker function run in a separate thread to save an image to disk.
        """
        try:
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

    def is_image_ready(self, path):
        """
        Check if the image at the given path is fully saved to disk.

        Parameters
        ----------
        path : str

        Returns
        -------
        bool
            True if the image is saved, False otherwise
        """
        return path in self.saved_images

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
