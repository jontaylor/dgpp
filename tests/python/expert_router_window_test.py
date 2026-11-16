#!/usr/bin/env python3
"""Synthetic timeline checks for exact-route handoff attribution."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import struct
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('window',Path(__file__).resolve().parents[2]/'scripts/expert_router_window.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


class WindowTests(unittest.TestCase):
    def fixture(self,kernels,reads):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        p=Path(tmp.name);db=p/'timeline.sqlite';c=sqlite3.connect(db)
        c.executescript('''CREATE TABLE TARGET_INFO_SESSION_START_TIME(utcEpochNs INTEGER);
          INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES(1000000000);
          CREATE TABLE StringIds(id INTEGER,value TEXT);
          CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,streamId INTEGER,shortName INTEGER,gridX INTEGER,gridY INTEGER,graphNodeId INTEGER);
          CREATE TABLE OSRT_API(start INTEGER,end INTEGER,nameId INTEGER,globalTid INTEGER);''')
        names={name:i for i,name in enumerate({k[3] for k in kernels}|{'pread64','write'})}
        c.executemany('INSERT INTO StringIds VALUES(?,?)',[(v,k) for k,v in names.items()])
        for a,b,stream,name,tokens in kernels:
            c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?)',(a,b,stream,names[name],tokens, tokens,0))
        for t,layer in reads:
            c.execute('INSERT INTO OSRT_API VALUES(?,?,?,?)',(t+5,t+100,names['pread64'],99))
            c.execute('INSERT INTO OSRT_API VALUES(?,?,?,?)',(t+101,t+110,names['write'],99))
        c.commit();c.close()
        path=p/'reads.bin'
        path.write_bytes(b''.join(struct.pack('<8Q',10**9+t,10**9+t+100,layer,0,0,0,4096,0) for t,layer in reads))
        return db,path

    def test_graph_reroute_counted_once_and_input_boundary_preserved(self):
        db,p=self.fixture([(100,110,1,'mhc_dots_kernel',4),(120,140,1,'moe_router_dots_kernel',4),
          (150,500,1,'demand_kernel',4),(200,220,2,'moe_router_dots_kernel',4)],[(300,7)])
        r=w.analyze(db,[p],[]);rows=r['observations']
        self.assertEqual(len(rows),1)
        row=rows[0]
        self.assertEqual(row['kind'],'decode_graph_miss')
        self.assertEqual(row['input_to_router_start_ns'],10)
        self.assertEqual(row['selection_to_read_ns'],165)
        self.assertEqual(row['service_selection_to_read_ns'],85)

    def test_prefill_includes_final_selection_and_hash_override(self):
        db,p=self.fixture([(100,110,1,'rmsnorm_kernel',128),(120,140,1,'moe_router_dots_tiled_kernel',8),
          (150,160,1,'moe_router_select_kernel',128),(170,180,1,'moe_hash_routes_kernel',128),
          (400,500,1,'other',1)],[(300,2)])
        row=w.analyze(db,[p],[])['observations'][0]
        self.assertEqual(row['kind'],'prefill')
        self.assertTrue(row['hash'])
        self.assertEqual(row['router_ns'],60)
        self.assertEqual(row['selection_to_read_ns'],125)

    def test_next_route_bounds_read_attribution(self):
        db,p=self.fixture([(10,20,1,'rmsnorm_kernel',128),(30,40,1,'moe_router_dots_kernel',128),
          (100,110,1,'rmsnorm_kernel',128),(120,130,1,'moe_router_dots_kernel',128),
          (500,600,1,'other',1)],[(300,3)])
        rows=w.analyze(db,[p],[])['observations']
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['start'],120)

    def test_later_router_on_other_stream_owns_read(self):
        db,p=self.fixture([(10,20,1,'rmsnorm_kernel',5),(30,40,1,'moe_router_dots_kernel',5),
          (100,110,2,'rmsnorm_kernel',128),(120,130,2,'moe_router_dots_kernel',128),
          (500,600,2,'other',1)],[(300,3)])
        rows=w.analyze(db,[p],[])['observations']
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['start'],120)

    def test_graph_producer_can_be_on_different_virtual_stream(self):
        db,p=self.fixture([(100,110,1,'mhc_dots_kernel',4),(120,140,2,'moe_router_dots_kernel',4),
          (150,500,2,'demand_kernel',4),(200,220,3,'moe_router_dots_kernel',4)],[(300,7)])
        c=sqlite3.connect(db)
        c.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET graphNodeId=10 WHERE start=100")
        c.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET graphNodeId=12 WHERE start=120")
        c.commit();c.close()
        row=w.analyze(db,[p],[])['observations'][0]
        self.assertEqual(row['input_to_router_start_ns'],10)

    def test_supplied_routes_need_no_service_router(self):
        db,p=self.fixture([(100,110,1,'mhc_dots_kernel',4),(120,140,1,'moe_router_dots_kernel',4),
          (150,500,1,'demand_kernel',4)],[(180,7)])
        rows=w.analyze(db,[p],[],reuse_graph_routes=True)['observations']
        self.assertEqual(len(rows),1)
        self.assertTrue(rows[0]['reused_graph_routes'])
        self.assertEqual(rows[0]['selection_to_read_ns'],45)

    def test_worker_count_disagreement_rejected(self):
        db,p=self.fixture([(100,110,1,'rmsnorm_kernel',128),(120,140,1,'moe_router_dots_kernel',128)],[(300,3)])
        with self.assertRaises(ValueError):w.analyze(db,[p],[],expected_workers=8)


if __name__=='__main__':unittest.main()
