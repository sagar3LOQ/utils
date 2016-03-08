import timeit



#### class for time measurement
class timeMeasure:

    def __init__(self):
        self.begin = 0
        self.end = 0
        self.elapsedTime = 0

    def start(self):
        self.begin = timeit.default_timer()
        self.end = 0
        self.elapsedTime = 0

    def stop(self):
        self.end = timeit.default_timer()
        self.elapsedTime += (self.end - self.begin)
        return self.elapsedTime

    def pause(self):
        self.end = timeit.default_timer()
        self.elapsedTime += (self.end - self.begin)
        return self.elapsedTime

    def reset(self):
        self.begin = 0
        self.end = 0
        self.elapsedTime = 0

    def resume(self):
	if(self.elapsedTime>0):
	        self.begin = timeit.default_timer()
	        self.end = 0
        return

    def getTime(self):
        return self.elapsedTime


