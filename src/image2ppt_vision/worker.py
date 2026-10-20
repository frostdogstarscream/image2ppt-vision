import multiprocessing as mp
import threading
import time
from .storage import Store, read_json


def child(root,connection):
    from .engine import execute
    while True:
        id=connection.recv()
        if id is None:break
        try:execute(root,id)
        except Exception as e:
            store=Store(root);store.seal(id);store.update(id,"FAILED",error_type=type(e).__name__)
        connection.send(id)


class Worker:
    def __init__(self,store):
        self.store=store;self.stop_event=threading.Event();self.thread=None;self.process=None;self.connection=None

    def start(self):
        self.store.recover()
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()

    def loop(self):
        while not self.stop_event.is_set():
            id=self.store.next_job()
            if not id:self.stop_event.wait(.2);continue
            self.run(id)

    def run(self,id):
        self.store.update(id,"RUNNING")
        r=read_json(self.store.path("jobs",id)/"request.json")
        if self.process is None or not self.process.is_alive():
            if self.connection:self.connection.close()
            self.connection,remote=mp.get_context("spawn").Pipe()
            self.process=mp.get_context("spawn").Process(target=child,args=(str(self.store.root),remote))
            self.process.start();remote.close()
        p=self.process;self.connection.send(id);start=time.monotonic();done=False
        while p.is_alive() and not self.stop_event.is_set() and time.monotonic()-start<r["timeout_seconds"]:
            if self.connection.poll(.2):
                try:done=self.connection.recv()==id
                except EOFError:pass
                break
        if not done and p.is_alive():
            p.terminate();p.join(5)
            if p.is_alive():p.kill();p.join()
            self.store.seal(id)
            self.store.update(id,"INTERRUPTED" if self.stop_event.is_set() else "FAILED",error_type="Shutdown" if self.stop_event.is_set() else "Timeout")
        elif not done and self.store.status(id)["status"]=="RUNNING":
            self.store.seal(id);self.store.update(id,"FAILED",error_type="WorkerExit")

    def stop(self):
        self.stop_event.set()
        if self.thread:self.thread.join(10)
        if self.process and self.process.is_alive():
            try:self.connection.send(None)
            except (OSError,EOFError):pass
            self.process.join(3)
            if self.process.is_alive():self.process.terminate();self.process.join(5)
        if self.connection:self.connection.close()
