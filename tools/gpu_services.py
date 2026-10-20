"""Only the nine explicitly authorized containers; never deletes anything."""
import argparse
import json
import subprocess
from pathlib import Path

NAMES=[f"pengcheng-platform-t1-{i:02}-1" for i in range(1,9)]+["pengcheng-platform-t2-1"]

def run(*args):return subprocess.check_output(args,text=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=["snapshot","stop","restore"]);p.add_argument("--record",required=True);a=p.parse_args()
    record=Path(a.record)
    if a.action=="snapshot":
        raw=json.loads(run("docker","inspect",*NAMES))
        data=[{"name":x["Name"].lstrip("/"),"id":x["Id"],"image":x["Config"]["Image"],"running":x["State"]["Running"],"restart":x["HostConfig"]["RestartPolicy"]} for x in raw]
        with record.open("x",encoding="utf-8") as f:json.dump(data,f,indent=2)
        print("saved",len(data));return
    records=json.loads(record.read_text())
    if sorted(r["name"] for r in records)!=sorted(NAMES):raise ValueError("snapshot scope mismatch")
    for r in records:
        current=json.loads(run("docker","inspect",r["name"]))[0]
        if current["Id"]!=r["id"]:raise ValueError("container identity changed")
    for r in records:
        if a.action=="stop":
            run("docker","update","--restart=no",r["id"])
            if r["running"]:run("docker","stop","--time","30",r["id"])
        else:
            policy=r["restart"]["Name"] or "no";n=r["restart"]["MaximumRetryCount"]
            if policy=="on-failure" and n:policy+=":"+str(n)
            run("docker","update","--restart="+policy,r["id"])
            if r["running"]:run("docker","start",r["id"])
        print(a.action,r["name"],flush=True)

if __name__=="__main__":main()
