#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FOD_sistem.py - sistem utuh, tinggal jalankan:  python FOD_sistem.py

Deteksi FOD (YOLOv8n) + jarak stereo + geolokasi (GPS + IMU via ESP32).
Alur data:  program ini --MQTT--> broker CloudAMQP --> jembatan (thread di program ini) --> Firestore --> website
Data CSV otomatis disimpan tiap jarak (maju_cm) berubah LOG_STEP_CM (default 5 cm),
dan pada saat yang sama dikirim lewat MQTT.

Butuh:  pip install paho-mqtt firebase-admin pyserial
        + file serviceAccountKey.json (Firebase Console > Project settings > Service accounts >
          Generate new private key) di folder yang sama.

Tombol: q = keluar | r = mulai ulang urutan simpan | s = simpan screenshot |
        g = on/off tanah (mode otomatis) | n = sesi rekaman baru
"""

import argparse
import base64
import csv
import datetime as _dt
import json
import math
import os
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from ultralytics import YOLO

# ============================ MQTT (CloudAMQP) ============================
# Topik: <prefix>/session, <prefix>/track, <prefix>/detections
# Sebaiknya password diisi lewat environment variable FOD_MQTT_PASS.
MQTT_HOST = os.environ.get("FOD_MQTT_HOST", "albatross.rmq.cloudamqp.com")
MQTT_TLS = os.environ.get("FOD_MQTT_TLS", "1") == "1"            # 1 = port 8883 (terenkripsi)
MQTT_PORT = int(os.environ.get("FOD_MQTT_PORT", "8883" if MQTT_TLS else "1883"))
MQTT_USER = os.environ.get("FOD_MQTT_USER", "bccpuelu:bccpuelu")
MQTT_PASS = os.environ.get("FOD_MQTT_PASS", "qyD70yw9bMfTGFJBM7qzBYezCPkzObze")
MQTT_PREFIX = os.environ.get("FOD_MQTT_PREFIX", "fod-stereo-polinema")
MQTT_TRACK_EVERY_S = 2.0                                         # jeda kirim titik jalur

FS_KEY = os.environ.get("FOD_SERVICE_KEY", "serviceAccountKey.json")
FS_COL_DET, FS_COL_SES = "fod_detections", "fod_sessions"


def encode_image(img, max_w=480, quality=70):
    """BGR image -> string base64 JPEG (tanpa prefix data URI)."""
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


class MqttPublisher:
    """Mengirim data sensor ke broker MQTT (QoS 1, thread jaringan sendiri)."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.n_ok = self.n_fail = self.n_sent = 0
        self.session, self.session_started = None, False
        self._last_track, self._seq, self._last_info = 0.0, 0, None
        if not enabled:
            return
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                  client_id=f"{MQTT_PREFIX}-jetson-{os.getpid()}")
        self.client.username_pw_set(MQTT_USER, MQTT_PASS)
        if MQTT_TLS:
            self.client.tls_set()
        self.client.reconnect_delay_set(1, 30)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        self.client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=30)
        self.client.loop_start()
        print(f"[MQTT] menyambung ke {MQTT_HOST}:{MQTT_PORT} (TLS={'ya' if MQTT_TLS else 'tidak'}) "
              f"prefix={MQTT_PREFIX}")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        print("[MQTT] tersambung" if not rc.is_failure else f"[MQTT] gagal tersambung: {rc}")

    def _on_disconnect(self, client, userdata, flags, rc, properties=None):
        print(f"[MQTT] terputus ({rc}); pesan QoS 1 diantre & dikirim ulang saat tersambung")

    def _on_publish(self, client, userdata, mid, rc=None, properties=None):
        self.n_ok += 1

    def _pub(self, sub, payload):
        info = self.client.publish(f"{MQTT_PREFIX}/{sub}", json.dumps(payload), qos=1)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            self.n_fail += 1
        self.n_sent += 1
        self._last_info = info

    def start_session(self, lat=None, lon=None):
        if not self.enabled:
            return
        self.session = time.strftime("S%Y%m%d_%H%M%S")
        self.session_started = True
        self._last_track = time.time()
        self._pub("session", {"session": self.session, "state": "start",
                              "lat": lat, "lon": lon, "timestamp": time.time()})
        print(f"[MQTT] sesi {self.session} dimulai")

    def track(self, lat, lon):
        if not (self.enabled and self.session_started) or lat is None:
            return
        if time.time() - self._last_track >= MQTT_TRACK_EVERY_S:
            self._last_track = time.time()
            self._pub("track", {"session": self.session, "lat": lat, "lon": lon,
                                "timestamp": time.time()})

    def end_session(self, lat=None, lon=None):
        if not (self.enabled and self.session_started):
            return
        self._pub("session", {"session": self.session, "state": "end",
                              "lat": lat, "lon": lon, "timestamp": time.time()})
        self.session_started = False
        print(f"[MQTT] sesi {self.session} selesai")

    def push(self, rec):
        if not self.enabled:
            return
        self._seq += 1
        rec = dict(rec)
        rec["id"] = f"{int(time.time() * 1000)}_{self._seq}"
        rec["session"] = self.session
        rec["timestamp"] = time.time()
        self._pub("detections", rec)

    def close(self):
        if not self.enabled:
            return
        try:
            if self._last_info is not None:
                self._last_info.wait_for_publish(timeout=5)   # pastikan pesan terakhir terkirim
        except Exception:
            pass
        self.client.loop_stop()
        self.client.disconnect()


def _fs_dt(v):
    if v is None:
        return _dt.datetime.now(_dt.timezone.utc)
    if isinstance(v, (int, float)):
        return _dt.datetime.fromtimestamp(v / 1000 if v > 1e12 else v, _dt.timezone.utc)
    d = _dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)


class FirestoreBridge:
    """Subscriber MQTT (client & thread sendiri): ambil data dari broker -> simpan ke Firestore."""

    def __init__(self, enabled=True):
        self.enabled = False
        self.n_done = 0
        self.current = None
        if not enabled:
            return
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore
        except ImportError:
            print("[BRIDGE] firebase-admin belum terpasang (pip install firebase-admin) "
                  "-> data hanya sampai broker, TIDAK masuk Firestore")
            return
        if not os.path.isfile(FS_KEY):
            print(f"[BRIDGE] file kunci '{FS_KEY}' tidak ditemukan -> data hanya sampai broker, "
                  "TIDAK masuk Firestore")
            return
        if not firebase_admin._apps:
            firebase_admin.initialize_app(credentials.Certificate(FS_KEY))
        self._fs = firestore
        self.db = firestore.client()
        # client_id tetap + clean_session=False: pesan QoS 1 yang datang saat program mati tidak hilang
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                  client_id=f"{MQTT_PREFIX}-bridge", clean_session=False)
        self.client.username_pw_set(MQTT_USER, MQTT_PASS)
        if MQTT_TLS:
            self.client.tls_set()
        self.client.reconnect_delay_set(1, 30)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=30)
        self.client.loop_start()
        self.enabled = True
        print("[BRIDGE] aktif: pesan MQTT diteruskan ke Firestore")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if not rc.is_failure:
            client.subscribe([(f"{MQTT_PREFIX}/detections", 1), (f"{MQTT_PREFIX}/session", 1),
                              (f"{MQTT_PREFIX}/track", 1)])

    def _on_message(self, client, userdata, msg):
        try:
            self._handle(msg.topic.rsplit("/", 1)[-1], json.loads(msg.payload))
        except Exception as e:
            print(f"[BRIDGE] gagal memproses {msg.topic}: {e}")
        finally:
            self.n_done += 1

    def _handle(self, kind, m):
        if kind == "session":
            sid = m["session"]
            ref = self.db.collection(FS_COL_SES).document(sid)
            pt = {"lat": m.get("lat"), "lon": m.get("lon"),
                  "ts": int(_fs_dt(m.get("timestamp")).timestamp() * 1000)}
            if m.get("state") == "start":
                self.current = sid
                if not ref.get().exists:          # idempoten: start yang terkirim ulang tidak menghapus data
                    track = [{"lat": m["lat"], "lon": m["lon"]}] if m.get("lat") is not None else []
                    ref.set({"start": pt, "end": None, "track": track})
            elif m.get("state") == "end":
                ref.set({"end": pt}, merge=True)
                if self.current == sid:
                    self.current = None
        elif kind == "detections":
            doc = {"class": m.get("class"), "distance": m.get("distance"),
                   "lat": m.get("lat"), "lon": m.get("lon"), "imageUrl": m.get("imageUrl"),
                   "timestamp": _fs_dt(m.get("timestamp")),
                   "session": m.get("session") or self.current}
            col = self.db.collection(FS_COL_DET)
            (col.document(m["id"]) if m.get("id") else col.document()).set(doc)   # id sama = tidak dobel
        elif kind == "track":
            sid = m.get("session") or self.current
            if sid and m.get("lat") is not None:
                self.db.collection(FS_COL_SES).document(sid).set(
                    {"track": self._fs.ArrayUnion([{"lat": m["lat"], "lon": m["lon"]}])}, merge=True)

    def close(self, expect=0, timeout=8.0):
        """Tunggu sampai pesan yang baru dikirim selesai ditulis ke Firestore, lalu putus."""
        if not self.enabled:
            return
        t0 = time.time()
        while self.n_done < expect and time.time() - t0 < timeout:
            time.sleep(0.2)
        self.client.loop_stop()
        self.client.disconnect()


# ============================ KONFIGURASI ============================
WEIGHTS_PATH = r"D:\SKRIPSI_PROGRAM\Training_YOLOv8n\FOD_detection\weights\best.pt"
CALIB_PATH = r"D:\SKRIPSI_PROGRAM\stereo_calib.npz"

CAM_INDEX = 0
FRAME_W, FRAME_H = 2560, 720
CROP_FRAC = 0.60

DEPTH_SCALE = 1.0
DISP_OFFSET_PX = 5.3

CAM_HEIGHT_CM = 76.0
CAM_TILT_DEG = 35.0
GROUND_CHECK_EVERY = 15

# ---- auto-simpan ----
LOG_STEP_CM = 5.0                     # simpan tiap jarak berubah sebesar ini (cm)
SMOOTH_N = 5                          # median N frame supaya noise tidak memicu simpan palsu

# ---- data yang dikirim ----
FB_DISTANCE_DIV = 100.0               # distance dikirim = maju_cm / FB_DISTANCE_DIV (100 -> meter, 1 -> cm)
FB_IMG_WIDTH = 480                    # lebar gambar yang dikirim (px)
FB_IMG_QUALITY = 70                   # kualitas JPEG 1-100

# Stereo matching (SGBM)
DISP_SCALE = 0.5
NUM_DISP = 16 * 8
MIN_DISP = -16
BLOCK = 7
MIN_VALID_FRAC = 0.15
INNER_FRAC = 0.5
MAX_RANGE_CM = 600.0

# Estimasi bidang tanah (RANSAC)
GROUND_ROW_START = 0.40
GROUND_STRIDE = 6
RANSAC_ITERS = 150
RANSAC_THRESH_CM = 2.0
GROUND_MIN_INLIER_FRAC = 0.30
GROUND_EMA = 0.2

M_PER_DEG_LAT = 111320.0
# =====================================================================


# ============================ KALIBRASI & STEREO ============================
def valid_P(P1, P2):
    return bool(np.isfinite(P1).all() and np.isfinite(P2).all()
                and P1[0, 0] > 10 and P2[0, 0] > 10)


def load_calib(path, model_opt="auto"):
    if not os.path.isfile(path):
        raise SystemExit(f"File kalibrasi tidak ditemukan: {path}")
    d = np.load(path, allow_pickle=True)
    print("[INFO] file kalibrasi:", path)
    print("[INFO] key:", d.files)

    for k in ("mtx_left", "dist_left", "mtx_right", "dist_right", "R", "T"):
        if k not in d.files:
            raise SystemExit(f"Key '{k}' tidak ada di file kalibrasi")

    K1 = d["mtx_left"].astype(np.float64)
    K2 = d["mtx_right"].astype(np.float64)
    D1 = np.asarray(d["dist_left"], np.float64).reshape(-1, 1)
    D2 = np.asarray(d["dist_right"], np.float64).reshape(-1, 1)
    R = d["R"].astype(np.float64)
    T = d["T"].astype(np.float64).reshape(3, 1)

    for nama, M in [("K1", K1), ("K2", K2), ("D1", D1), ("D2", D2), ("R", R), ("T", T)]:
        if not np.isfinite(M).all():
            raise SystemExit(f"{nama} mengandung NaN/inf -> kalibrasi rusak, kalibrasi ulang")
    if K1[0, 0] < 10 or K2[0, 0] < 10:
        raise SystemExit(f"Focal length K tidak valid: {K1[0, 0]}, {K2[0, 0]}")

    flag_fish = ("is_fisheye" in d.files) and bool(d["is_fisheye"])
    if model_opt == "auto":
        model = "fisheye" if (D1.size == 4 and flag_fish) else "pinhole"
    else:
        model = model_opt
    if model == "fisheye" and (D1.size != 4 or D2.size != 4):
        raise SystemExit(f"Model fisheye butuh 4 koefisien distorsi, file berisi {D1.size}. "
                         "Gunakan --model pinhole atau kalibrasi ulang.")
    if model == "pinhole" and D1.size not in (4, 5, 8, 12, 14):
        raise SystemExit(f"Jumlah koefisien distorsi {D1.size} tidak didukung model pinhole")

    if "img_shape" in d.files:
        print("[INFO] img_shape:", d["img_shape"], "(harus = ukuran satu mata:",
              (FRAME_W // 2, FRAME_H), ")")

    for k in ("rms_left", "rms_right", "rms_stereo", "reprojection_left", "reprojection_right"):
        if k in d.files:
            try:
                print(f"[INFO] {k} = {float(d[k]):.4f}")
            except (TypeError, ValueError):
                pass

    saved = None
    if all(k in d.files for k in ("R1", "R2", "P1", "P2")):
        saved = (d["R1"].astype(np.float64), d["R2"].astype(np.float64),
                 d["P1"].astype(np.float64), d["P2"].astype(np.float64))
        if not valid_P(saved[2], saved[3]):
            print(f"[PERINGATAN] P1/P2 di .npz tidak valid (f = {saved[2][0, 0]:.1f} px) "
                  "-> rektifikasi dihitung ulang dari K, D, R, T")
            saved = None

    print(f"[INFO] model lensa: {model} | f_kiri = {K1[0, 0]:.1f} px | D1 = {D1.ravel()} | "
          f"T = {T.ravel()}")
    return K1, D1, K2, D2, R, T, saved, model


def to_cm_factor(T):
    b = float(np.linalg.norm(T))
    if b < 1.0:
        return 100.0
    if b < 20.0:
        return 1.0
    return 0.1


class StereoRig:
    def __init__(self, calib_path, model_opt="auto"):
        K1, D1, K2, D2, R, T, saved, model = load_calib(calib_path, model_opt)
        self.model = model
        self.unit = to_cm_factor(T)
        T = T * self.unit
        self.baseline = float(np.linalg.norm(T))
        self.K1, self.D1, self.K2, self.D2, self.R, self.T = K1, D1, K2, D2, R, T
        self.saved = saved
        self.size = (FRAME_W // 2, FRAME_H)
        self._using_saved = False
        self.build_maps()

        self.min_d = int(round(MIN_DISP * DISP_SCALE))
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=self.min_d, numDisparities=NUM_DISP, blockSize=BLOCK,
            P1=8 * BLOCK * BLOCK, P2=32 * BLOCK * BLOCK,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=2,
            preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def _make_maps(self, R1, R2, P1, P2):
        if self.model == "fisheye":
            m1 = cv2.fisheye.initUndistortRectifyMap(self.K1, self.D1, R1, P1, self.size, cv2.CV_16SC2)
            m2 = cv2.fisheye.initUndistortRectifyMap(self.K2, self.D2, R2, P2, self.size, cv2.CV_16SC2)
        else:
            m1 = cv2.initUndistortRectifyMap(self.K1, self.D1, R1, P1, self.size, cv2.CV_16SC2)
            m2 = cv2.initUndistortRectifyMap(self.K2, self.D2, R2, P2, self.size, cv2.CV_16SC2)
        return m1, m2

    def _valid_ratio(self, maps):
        w, h = self.size
        ones = np.full((h, w), 255, np.uint8)
        out = cv2.remap(ones, maps[0], maps[1], cv2.INTER_LINEAR)
        x0, y0 = int(w * (1 - CROP_FRAC) / 2), int(h * (1 - CROP_FRAC) / 2)
        return float((out[y0:h - y0, x0:w - x0] > 250).mean())

    def _recompute_candidates(self):
        size = self.size
        cands = []
        if self.model == "fisheye":
            for bal, fov in [(0.0, 1.0), (0.3, 1.0), (0.0, 0.8), (0.6, 1.0), (1.0, 1.0)]:
                try:
                    R1, R2, P1, P2, _ = cv2.fisheye.stereoRectify(
                        self.K1, self.D1, self.K2, self.D2, size, self.R, self.T,
                        flags=cv2.CALIB_ZERO_DISPARITY, newImageSize=size,
                        balance=bal, fov_scale=fov)
                    cands.append((f"hitung ulang fisheye (balance={bal}, fov={fov})", R1, R2, P1, P2))
                except cv2.error as e:
                    print(f"[INFO] fisheye.stereoRectify balance={bal} fov={fov} gagal: {e}")
            try:
                zero = np.zeros((5, 1), np.float64)
                R1, R2, _, _, _, _, _ = cv2.stereoRectify(
                    self.K1, zero, self.K2, zero, size, self.R, self.T,
                    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
                w, h = size
                for fs in (1.0, 0.8, 0.6):
                    f = float(self.K1[0, 0]) * fs
                    P1 = np.array([[f, 0, w / 2.0, 0], [0, f, h / 2.0, 0], [0, 0, 1, 0]], np.float64)
                    P2 = P1.copy()
                    P2[0, 3] = -f * self.baseline
                    cands.append((f"manual fisheye (R dari stereoRectify, f={fs}*K)", R1, R2, P1, P2))
            except cv2.error as e:
                print(f"[INFO] kandidat manual fisheye gagal: {e}")
        else:
            for alpha in [0, 0.3, 1, -1]:
                try:
                    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
                        self.K1, self.D1, self.K2, self.D2, size, self.R, self.T,
                        flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
                    cands.append((f"hitung ulang pinhole (alpha={alpha})", R1, R2, P1, P2))
                except cv2.error as e:
                    print(f"[INFO] stereoRectify alpha={alpha} gagal: {e}")
        return cands

    def build_maps(self, skip_saved=False):
        size = self.size
        cands = []
        if self.saved is not None and not skip_saved:
            R1, R2, P1, P2 = self.saved
            cands.append(("tersimpan di .npz", R1, R2, P1, P2))
        cands += self._recompute_candidates()

        best, best_ratio = None, -1.0
        for name, R1, R2, P1, P2 in cands:
            if not valid_P(P1, P2):
                print(f"[CEK] {name}: ditolak (f tidak valid)")
                continue
            m1, m2 = self._make_maps(R1, R2, P1, P2)
            ratio = min(self._valid_ratio(m1), self._valid_ratio(m2))
            print(f"[CEK] {name}: f = {P1[0, 0]:.1f} px, area terisi = {ratio * 100:.1f}%")
            if ratio > best_ratio:
                best, best_ratio = (name, m1, m2, P1), ratio
            if ratio >= 0.98:
                break
        if best is None:
            raise SystemExit("Tidak ada rektifikasi yang valid. Cek K, D, R, T dan --model "
                             "(pinhole/fisheye) di file kalibrasi.")
        if best_ratio < 0.9:
            print(f"[PERINGATAN] area terisi hanya {best_ratio * 100:.1f}% -> cek model lensa "
                  f"(--model), resolusi kamera = {FRAME_W}x{FRAME_H}, dan file kalibrasi")

        name, self.map1, self.map2, P1 = best
        self._using_saved = name.startswith("tersimpan")
        w, h = size
        self.x0 = int(w * (1 - CROP_FRAC) / 2)
        self.y0 = int(h * (1 - CROP_FRAC) / 2)
        self.x1, self.y1 = w - self.x0, h - self.y0
        self.f = float(P1[0, 0])
        self.cx = float(P1[0, 2]) - self.x0
        self.cy = float(P1[1, 2]) - self.y0
        print(f"[INFO] rektifikasi dipakai: {name} | baseline = {self.baseline:.2f} cm | "
              f"f = {self.f:.1f} px | crop = {self.x1 - self.x0}x{self.y1 - self.y0}")
        if name.startswith("manual fisheye"):
            print("[PERINGATAN] rektifikasi fisheye resmi gagal (f=0), dipakai rektifikasi cadangan. "
                  "Skala jarak BELUM tervalidasi.")

    def maps_saved_in_use(self):
        return self._using_saved

    def rectify(self, frame):
        left = frame[:, :FRAME_W // 2]
        right = frame[:, FRAME_W // 2:]
        l = cv2.remap(left, self.map1[0], self.map1[1], cv2.INTER_LINEAR)
        r = cv2.remap(right, self.map2[0], self.map2[1], cv2.INTER_LINEAR)
        l = l[self.y0:self.y1, self.x0:self.x1]
        r = r[self.y0:self.y1, self.x0:self.x1]
        return l, r

    def disparity(self, l_bgr, r_bgr):
        gl = cv2.cvtColor(l_bgr, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(r_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gl.shape
        if DISP_SCALE != 1.0:
            sz = (int(w * DISP_SCALE), int(h * DISP_SCALE))
            gl = cv2.resize(gl, sz, interpolation=cv2.INTER_AREA)
            gr = cv2.resize(gr, sz, interpolation=cv2.INTER_AREA)
        d = self.sgbm.compute(gl, gr).astype(np.float32) / 16.0
        d[d <= self.min_d - 0.5] = np.nan
        if DISP_SCALE != 1.0:
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST) / DISP_SCALE
        d = d + DISP_OFFSET_PX
        with np.errstate(invalid="ignore"):
            d[d <= 0.5] = np.nan
        return d

    def point_3d(self, u, v, disp):
        z = self.f * self.baseline / disp * DEPTH_SCALE
        return np.array([(u - self.cx) * z / self.f,
                         (v - self.cy) * z / self.f,
                         z], dtype=np.float64)


def box_disparity(disp, box):
    x1, y1, x2, y2 = box
    h, w = disp.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 2 or bh < 2:
        return None
    mx = int(bw * (1 - INNER_FRAC) / 2)
    my = int(bh * (1 - INNER_FRAC) / 2)
    roi = disp[y1 + my:y2 - my, x1 + mx:x2 - mx]
    if roi.size == 0:
        roi = disp[y1:y2, x1:x2]
    v = roi[np.isfinite(roi)]
    if v.size < max(5, MIN_VALID_FRAC * roi.size):
        return None
    return float(np.median(v))


# ============================ BIDANG TANAH ============================
class GroundPlane:
    def __init__(self):
        self.n = None
        self.d0 = None
        self.manual = False
        self.rng = np.random.default_rng(0)

    def set_manual(self, height_cm, tilt_deg):
        t = math.radians(tilt_deg)
        self.n = np.array([0.0, -math.cos(t), -math.sin(t)])
        self.d0 = -float(height_cm)
        self.manual = True

    def update(self, disp, rig, exclude_boxes):
        h, w = disp.shape
        ys, xs = np.mgrid[int(h * GROUND_ROW_START):h:GROUND_STRIDE, 0:w:GROUND_STRIDE]
        d = disp[ys, xs]
        ok = np.isfinite(d) & (d > 1.0)
        for (bx1, by1, bx2, by2) in exclude_boxes:
            ok &= ~((xs >= bx1 - 5) & (xs <= bx2 + 5) & (ys >= by1 - 5) & (ys <= by2 + 5))
        d, xs, ys = d[ok], xs[ok].astype(np.float64), ys[ok].astype(np.float64)
        z = rig.f * rig.baseline / d * DEPTH_SCALE
        keep = z < MAX_RANGE_CM
        z, xs, ys = z[keep], xs[keep], ys[keep]
        if z.size < 200:
            return
        P = np.stack([(xs - rig.cx) * z / rig.f, (ys - rig.cy) * z / rig.f, z], axis=1)

        best_in, best_mask = 0, None
        for _ in range(RANSAC_ITERS):
            idx = self.rng.choice(len(P), 3, replace=False)
            p0, p1, p2 = P[idx]
            n = np.cross(p1 - p0, p2 - p0)
            nn = np.linalg.norm(n)
            if nn < 1e-6:
                continue
            n /= nn
            mask = np.abs((P - p0) @ n) < RANSAC_THRESH_CM
            c = int(mask.sum())
            if c > best_in:
                best_in, best_mask = c, mask
        if best_mask is None or best_in < GROUND_MIN_INLIER_FRAC * len(P):
            return

        Q = P[best_mask]
        c = Q.mean(axis=0)
        n = np.linalg.svd(Q - c)[2][-1]
        d0 = float(n @ c)
        if d0 > 0:
            n, d0 = -n, -d0

        if self.n is None:
            self.n, self.d0 = n, d0
        else:
            self.n = (1 - GROUND_EMA) * self.n + GROUND_EMA * n
            self.n /= np.linalg.norm(self.n)
            self.d0 = (1 - GROUND_EMA) * self.d0 + GROUND_EMA * d0

    @property
    def ready(self):
        return self.n is not None

    @property
    def height_cm(self):
        return -self.d0

    @property
    def tilt_deg(self):
        return float(np.degrees(np.arctan2(-self.n[2], -self.n[1])))

    def local_xy(self, P):
        n = self.n
        s = float(n @ P - self.d0)
        v = P - float(n @ P) * n
        z = np.array([0.0, 0.0, 1.0])
        fwd = z - float(z @ n) * n
        nf = np.linalg.norm(fwd)
        if nf < 1e-6:
            return None
        fwd /= nf
        right = np.cross(fwd, n)
        return float(v @ fwd), float(v @ right), s


# ============================ GPS + IMU (dari ESP32) ============================
def parse_line(line):
    line = line.strip()
    if not line:
        return None
    try:
        if line.startswith("{"):
            d = json.loads(line)
            hd = d["heading"] if "heading" in d else d["yaw"]
            return dict(lat=float(d["lat"]), lon=float(d["lon"]), heading=float(hd),
                        fix=int(d.get("fix", 1)), sats=int(d.get("sats", 0)),
                        hdop=float(d.get("hdop", 99.0)))
        if line.startswith("GEO,"):
            p = line.split(",")
            return dict(lat=float(p[1]), lon=float(p[2]), heading=float(p[3]),
                        fix=int(p[4]) if len(p) > 4 else 1,
                        sats=int(p[5]) if len(p) > 5 else 0,
                        hdop=float(p[6]) if len(p) > 6 else 99.0)
    except (ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError):
        return None
    return None


class GeoReader:
    def __init__(self, port=None, baud=115200, sim=None):
        self.lock = threading.Lock()
        self.latest, self.t = None, 0.0
        self.n_ok, self.n_bad = 0, 0
        self.sim = sim is not None
        self.running = False
        self.enabled = self.sim or bool(port)
        if self.sim:
            lat, lon, hd = sim
            self.latest = dict(lat=lat, lon=lon, heading=hd, fix=1, sats=0, hdop=0.0)
        elif port:
            try:
                import serial
            except ImportError:
                raise SystemExit("pyserial belum terpasang: pip install pyserial")
            try:
                self.ser = serial.Serial(port, baud, timeout=1)
            except Exception as e:
                raise SystemExit(f"Port serial {port} tidak bisa dibuka: {e}")
            self.running = True
            threading.Thread(target=self._run, daemon=True).start()
            print(f"[GEO] membaca serial {port} @ {baud}")

    def _run(self):
        while self.running:
            try:
                raw = self.ser.readline().decode("utf-8", errors="ignore")
            except Exception as e:
                print(f"[GEO] serial error: {e}")
                time.sleep(1.0)
                continue
            d = parse_line(raw)
            if d is None:
                if raw.strip() and not raw.startswith("#"):
                    self.n_bad += 1
                continue
            with self.lock:
                self.latest, self.t = d, time.time()
                self.n_ok += 1

    def get(self):
        with self.lock:
            if self.latest is None:
                return None, None
            age = 0.0 if self.sim else time.time() - self.t
            return dict(self.latest), age

    def close(self):
        self.running = False


def find_geo_port(baud, listen_s=4.0):
    try:
        import serial
        from serial.tools import list_ports
    except ImportError:
        print("[GEO] pyserial belum terpasang (pip install pyserial) -> geolokasi nonaktif")
        return None
    ports = [p.device for p in list_ports.comports()]
    if not ports:
        print("[GEO] tidak ada port serial terdeteksi")
        return None
    print(f"[GEO] mencari ESP32 di port: {ports}")
    for dev in ports:
        try:
            with serial.Serial(dev, baud, timeout=0.5) as s:
                t0 = time.time()
                while time.time() - t0 < listen_s:
                    raw = s.readline().decode("utf-8", errors="ignore")
                    if parse_line(raw) is not None:
                        print(f"[GEO] ESP32 ditemukan di {dev}")
                        return dev
        except Exception:
            continue
    print("[GEO] ESP32 tidak ditemukan (tidak ada data format GEO di port mana pun)")
    return None


def offset_to_latlon(lat0, lon0, d_north_m, d_east_m):
    lat = lat0 + d_north_m / M_PER_DEG_LAT
    lon = lon0 + d_east_m / (M_PER_DEG_LAT * math.cos(math.radians(lat0)))
    return lat, lon


def draw_label(img, text, org, color):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# ============================ MAIN ============================
def main():
    global DEPTH_SCALE, DISP_OFFSET_PX
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=WEIGHTS_PATH)
    ap.add_argument("--calib", default=CALIB_PATH)
    ap.add_argument("--model", default="auto", choices=["auto", "pinhole", "fisheye"])
    ap.add_argument("--source", default=str(CAM_INDEX))
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--log", default="auto")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--step-cm", type=float, default=LOG_STEP_CM)
    ap.add_argument("--smooth-n", type=int, default=SMOOTH_N)
    ap.add_argument("--cam-height", type=float, default=CAM_HEIGHT_CM)
    ap.add_argument("--tilt", type=float, default=CAM_TILT_DEG)
    ap.add_argument("--auto-ground", action="store_true")
    ap.add_argument("--no-ground", action="store_true")
    ap.add_argument("--depth-scale", type=float, default=DEPTH_SCALE)
    ap.add_argument("--disp-offset", type=float, default=DISP_OFFSET_PX)
    ap.add_argument("--geo-port", default="COM3")
    ap.add_argument("--geo-baud", type=int, default=115200)
    ap.add_argument("--geo-sim", default=None)
    ap.add_argument("--geo-max-age", type=float, default=1.0)
    ap.add_argument("--declination", type=float, default=0.0)
    ap.add_argument("--yaw-offset", type=float, default=0.0)
    # ---- MQTT / Firestore ----
    ap.add_argument("--no-firebase", action="store_true", help="nonaktifkan pengiriman MQTT & Firestore")
    ap.add_argument("--fb-allow-no-gps", action="store_true", default=True,
                    help="tetap kirim walau lat/lon kosong (GPS belum fix)")
    args = ap.parse_args()
    DEPTH_SCALE = args.depth_scale
    DISP_OFFSET_PX = args.disp_offset

    print(f"[INFO] {Path(__file__).name} | deteksi + stereo + geolokasi + MQTT")
    print(f"[INFO] koreksi disparity = +{DISP_OFFSET_PX:.2f} px | depth-scale = {DEPTH_SCALE:.3f}")
    print(f"[INFO] auto-simpan tiap jarak berubah {args.step_cm:.1f} cm (median {args.smooth_n} frame)")
    rig = StereoRig(args.calib, args.model)
    if args.recompute and rig.saved is not None:
        rig.build_maps(skip_saved=True)
    model = YOLO(args.weights)

    # ---- bidang tanah ----
    ground = GroundPlane()
    gcheck = GroundPlane()
    if args.no_ground:
        manual, use_ground = False, False
        mode_txt = "TANPA koreksi (kamera dianggap datar)"
    elif args.auto_ground:
        manual, use_ground = False, True
        mode_txt = "otomatis dari tanah (RANSAC)"
    else:
        manual, use_ground = True, False
        ground.set_manual(args.cam_height, args.tilt)
        mode_txt = f"manual (H={args.cam_height:.0f} cm, tilt={args.tilt:.0f} deg)"
    print(f"[INFO] mode H/tilt: {mode_txt}")

    # ---- GPS + IMU ----
    sim = None
    if args.geo_sim:
        try:
            a, b, c = [float(x) for x in args.geo_sim.split(",")]
            sim = (a, b, c)
        except ValueError:
            raise SystemExit("--geo-sim harus berformat lat,lon,heading")
    port = args.geo_port
    if sim is None:
        if port == "auto":
            port = find_geo_port(args.geo_baud)
        elif port and port.lower() == "none":
            port = None
    else:
        port = None
    geo = GeoReader(port, args.geo_baud, sim)
    if not geo.enabled:
        print("[INFO] geolokasi nonaktif (ESP32 tidak ditemukan): hanya jarak lokal.")

    # ---- MQTT (upload sensor) + jembatan ke Firestore ----
    fb = MqttPublisher(enabled=not args.no_firebase)
    bridge = FirestoreBridge(enabled=not args.no_firebase)
    if fb.enabled and not geo.enabled and not args.fb_allow_no_gps:
        print("[MQTT] PERINGATAN: GPS nonaktif -> lat/lon kosong, data TIDAK dikirim "
              "(pakai --fb-allow-no-gps atau --geo-sim untuk menguji)")

    # ---- kamera ----
    src = int(args.source) if args.source.isdigit() else args.source
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(src, backend) if isinstance(src, int) else cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        raise SystemExit("Kamera/video tidak bisa dibuka")
    for _ in range(15):
        cap.read()
    checked = False
    warned_size = False

    writer, logf = None, None
    log_path = None
    if not args.no_log:
        if args.log == "auto":
            os.makedirs("hasil_fod", exist_ok=True)
            log_path = os.path.join("hasil_fod", f"fod_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        else:
            log_path = args.log
    if log_path:
        print(f"[INFO] hasil disimpan ke: {log_path}")
        logf = open(log_path, "w", newline="")
        writer = csv.writer(logf)
        writer.writerow(["t", "kelas", "conf", "disp_px", "X_cm", "Y_cm", "Z_cm", "D3D_cm",
                         "maju_cm", "kanan_cm", "tinggi_obj_cm", "mode_tanah",
                         "H_est_cm", "tilt_est_deg",
                         "lat_platform", "lon_platform", "heading_deg", "gps_fix", "sats", "hdop",
                         "umur_gps_s", "lat_objek", "lon_objek"])

    # ---- auto-simpan tiap jarak berubah ----
    hist = deque(maxlen=max(1, args.smooth_n))
    last_saved = None
    n_saved = 0

    last_pos = None                       # posisi GPS platform terakhir (untuk titik akhir sesi)
    t_prev, fps = time.time(), 0.0
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
            if not warned_size:
                print(f"[PERINGATAN] frame {frame.shape[1]}x{frame.shape[0]} != "
                      f"{FRAME_W}x{FRAME_H}, di-resize")
                warned_size = True
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        left, right = rig.rectify(frame)

        if not checked:
            checked = True
            raw_mean = float(frame.mean())
            rect_mean = float(left.mean())
            print(f"[DIAG] frame mentah {frame.shape[1]}x{frame.shape[0]}, rata-rata piksel = {raw_mean:.1f} | "
                  f"hasil rektifikasi (kiri) = {rect_mean:.1f}")
            if raw_mean < 3:
                print("[DIAG] Frame mentah hitam -> cek indeks kamera / program lain yang memakai kamera")
            elif rect_mean < 3 and rig.saved is not None and rig.maps_saved_in_use():
                print("[DIAG] Rektifikasi hitam -> beralih ke rektifikasi hitung ulang")
                rig.build_maps(skip_saved=True)
                left, right = rig.rectify(frame)

        if args.debug and not args.no_show:
            cv2.imshow("RAW (debug)", cv2.resize(frame, (FRAME_W // 3, FRAME_H // 3)))

        disp = rig.disparity(left, right)

        res = model.predict(left, imgsz=args.imgsz, conf=args.conf,
                            device=args.device, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy().astype(int) if len(res.boxes) else []
        confs = res.boxes.conf.cpu().numpy() if len(res.boxes) else []
        clss = res.boxes.cls.cpu().numpy().astype(int) if len(res.boxes) else []

        if use_ground:
            ground.update(disp, rig, [tuple(b) for b in boxes])
        elif manual and frame_i % GROUND_CHECK_EVERY == 0:
            gcheck.update(disp, rig, [tuple(b) for b in boxes])
        frame_i += 1
        gest = gcheck if manual else ground

        gdata, gage = geo.get() if geo.enabled else (None, None)
        gps_ok = (gdata is not None and gdata["fix"] >= 1 and gage is not None
                  and gage <= args.geo_max_age)
        psi = None
        if gps_ok:
            psi = (gdata["heading"] + args.yaw_offset + args.declination) % 360.0

        # ---- sesi rekaman: mulai saat GPS fix pertama (atau langsung bila GPS nonaktif) ----
        if gps_ok:
            last_pos = (gdata["lat"], gdata["lon"])
        if fb.enabled:
            if not fb.session_started:
                if gps_ok or not geo.enabled:
                    fb.start_session(*(last_pos or (None, None)))
            elif gps_ok:
                fb.track(*last_pos)

        vis = left.copy()
        now = time.time()
        rows = []              # baris CSV semua deteksi valid pada frame ini
        recs = []              # record yang dikirim, paralel dengan rows
        best_conf, best_fwd = -1.0, None   # objek acuan pemicu simpan = conf tertinggi

        for b, cf, ci in zip(boxes, confs, clss):
            x1, y1, x2, y2 = [int(v) for v in b]
            dpx = box_disparity(disp, (x1, y1, x2, y2))
            name = model.names[int(ci)]
            if dpx is None:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 2)
                draw_label(vis, f"{name} {cf:.2f} | disp tak valid", (x1, max(15, y1 - 6)), (0, 165, 255))
                continue

            u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            P = rig.point_3d(u, v, dpx)
            D3 = float(np.linalg.norm(P))
            if P[2] > MAX_RANGE_CM:
                continue

            fwd = rgt = hobj = None
            mode_row = "tanpa_koreksi"
            if ground.ready:
                r = ground.local_xy(P)
                if r is not None:
                    fwd, rgt, hobj = r
                    mode_row = "manual" if ground.manual else "otomatis"
            if fwd is None:
                fwd, rgt = float(P[2]), float(P[0])

            lat_o = lon_o = None
            if gps_ok:
                f_m, r_m = fwd / 100.0, rgt / 100.0
                th = math.radians(psi)
                d_east = f_m * math.sin(th) + r_m * math.cos(th)
                d_north = f_m * math.cos(th) - r_m * math.sin(th)
                lat_o, lon_o = offset_to_latlon(gdata["lat"], gdata["lon"], d_north, d_east)

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            draw_label(vis, f"{name} {cf:.2f} | D={D3:.1f} cm  Z={P[2]:.1f}", (x1, max(15, y1 - 6)), (0, 255, 0))
            draw_label(vis, f"maju {fwd:.0f}  kanan {rgt:.0f} cm", (x1, y2 + 16), (0, 255, 255))
            if hobj is not None:
                draw_label(vis, f"tinggi {hobj:.1f} cm", (x1, y2 + 32), (255, 200, 0))
            if lat_o is not None:
                draw_label(vis, f"{lat_o:.6f}, {lon_o:.6f}", (x1, y2 + 48), (255, 255, 0))

            rows.append([
                f"{now:.3f}", name, f"{cf:.3f}", f"{dpx:.2f}",
                f"{P[0]:.2f}", f"{P[1]:.2f}", f"{P[2]:.2f}", f"{D3:.2f}",
                f"{fwd:.2f}", f"{rgt:.2f}", "" if hobj is None else f"{hobj:.2f}", mode_row,
                f"{gest.height_cm:.2f}" if gest.ready else "",
                f"{gest.tilt_deg:.2f}" if gest.ready else "",
                "" if gdata is None else f"{gdata['lat']:.7f}",
                "" if gdata is None else f"{gdata['lon']:.7f}",
                "" if psi is None else f"{psi:.1f}",
                "" if gdata is None else gdata["fix"],
                "" if gdata is None else gdata["sats"],
                "" if gdata is None else f"{gdata['hdop']:.1f}",
                "" if gage is None else f"{gage:.2f}",
                "" if lat_o is None else f"{lat_o:.7f}",
                "" if lon_o is None else f"{lon_o:.7f}"])
            # isi pesan MQTT: class, distance, imageUrl (diisi saat kirim), lat, lon
            recs.append({"class": name,
                         "distance": round(float(fwd) / FB_DISTANCE_DIV, 2),
                         "lat": lat_o, "lon": lon_o})
            if float(cf) > best_conf:
                best_conf, best_fwd = float(cf), fwd

        # ---- keputusan simpan: tiap jarak (median) berubah >= step-cm ----
        if best_fwd is None:
            hist.clear()                      # objek hilang -> mulai ulang penghalus
        else:
            hist.append(best_fwd)
            fwd_s = float(np.median(hist))
            siap = len(hist) == hist.maxlen
            pindah = last_saved is None or abs(fwd_s - last_saved) >= args.step_cm
            if siap and pindah:
                if writer:
                    for row in rows:
                        writer.writerow(row)
                    logf.flush()
                if fb.enabled:
                    kirim = [r for r in recs if r["lat"] is not None or args.fb_allow_no_gps]
                    if kirim:
                        img64 = encode_image(vis, max_w=FB_IMG_WIDTH, quality=FB_IMG_QUALITY)
                        for rec in kirim:
                            rec["imageUrl"] = img64
                            fb.push(rec)
                    elif recs:
                        print("[MQTT] dilewati: GPS belum fix, lat/lon kosong")
                last_saved = fwd_s
                n_saved += 1
                print(f"[SIMPAN #{n_saved}] maju = {fwd_s:.1f} cm")

        fps = 0.9 * fps + 0.1 / max(1e-6, now - t_prev)
        t_prev = now
        info = f"FPS {fps:.1f}"
        if ground.ready:
            info += f" | H~{ground.height_cm:.0f} cm tilt~{ground.tilt_deg:.0f} deg"
            if manual:
                info += " (manual)"
        if manual and gcheck.ready:
            info += f" | cek stereo: H~{gcheck.height_cm:.0f} tilt~{gcheck.tilt_deg:.0f}"
        info += f" | dispoff {DISP_OFFSET_PX:.1f}"
        info += f" | tersimpan {n_saved} | terakhir {('-' if last_saved is None else f'{last_saved:.0f} cm')}"
        if fb.enabled:
            info += f" | MQTT {fb.n_ok}/{fb.n_fail}"
        draw_label(vis, info, (10, 20), (255, 255, 255))
        if geo.enabled:
            if gdata is None:
                gtxt, gcol = "GPS: belum ada data", (0, 0, 255)
            elif gdata["fix"] < 1:
                gtxt, gcol = f"GPS: NO FIX ({gdata['sats']} sat)", (0, 0, 255)
            elif gage > args.geo_max_age:
                gtxt, gcol = f"GPS: data basi ({gage:.1f}s)", (0, 165, 255)
            else:
                gtxt = f"GPS ok | sat {gdata['sats']} hdop {gdata['hdop']:.1f} | hdg {psi:.0f}"
                gcol = (0, 255, 0)
            draw_label(vis, gtxt, (10, 40), gcol)
        else:
            draw_label(vis, "GPS: nonaktif", (10, 40), (180, 180, 180))

        if not args.no_show:
            cv2.imshow("FOD sistem: deteksi + jarak + geolokasi", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("g") and not manual:
                use_ground = not use_ground
            if k == ord("n") and fb.enabled:   # n = sesi rekaman baru
                fb.end_session(*(last_pos or (None, None)))
                fb.start_session(*(last_pos or (None, None)))
                last_saved = None
                hist.clear()
            if k == ord("s"):
                cv2.imwrite(f"cap_{int(now)}.png", vis)
            if k == ord("r"):
                last_saved = None
                hist.clear()
                print("[INFO] urutan simpan di-reset")

    cap.release()
    geo.close()
    fb.end_session(*(last_pos or (None, None)))
    fb.close()
    bridge.close(fb.n_sent)
    if logf:
        logf.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()