import time
from modules.Detect import Detect
from cv2 import waitKey
start_time = time.time()
detect = Detect(256).start()
print('Init finished')
print("--- took %s seconds ---" % (time.time() - start_time))
detect.enable()

while waitKey(27) != 27:
    time.sleep(1)

