from queue import Queue


class Events:
    def __init__(self):
        self.queue = Queue()
        self.done = False
        self.listeners = []

    def add(self, event, data=None, done=False):
        self.queue.put((event, data))
        self.done = done
        if done:
            for listener in self.listeners:
                listener()

    def on_done(self, callback):
        self.listeners.append(callback)

    def __iter__(self):
        while not self.done:
            yield self.queue.get()
            self.queue.task_done()
