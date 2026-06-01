from picamera2 import Picamera2
from PIL import Image
import time

picam2 = Picamera2()

config = picam2.create_still_configuration(main={"size": (1920, 1080)})
picam2.configure(config)

picam2.start()
print("Camera started, warming up...")

time.sleep(2)
image = picam2.capture_image()
image.save("cat_test.jpg")

picam2.stop()