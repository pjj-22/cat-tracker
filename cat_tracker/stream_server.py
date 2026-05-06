import threading
import queue
import json
import time
import cv2

try:
    from flask import Flask, Response
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    from flask_sock import Sock
    FLASK_SOCK_AVAILABLE = True
except ImportError:
    FLASK_SOCK_AVAILABLE = False


_HTML_HEAD = """\
<!doctype html>
<html>
<head>
  <title>Cat Tracker</title>
  <meta charset="utf-8">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0d0d0d; color: #ddd; font: 13px/1.4 monospace;
           display: flex; flex-direction: column; align-items: center; gap: 12px; padding: 16px; }
    img  { max-width: 100%; border: 1px solid #2a2a2a; }
    #status { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }
    .stat { background: #1a1a1a; border: 1px solid #2a2a2a; padding: 4px 12px; border-radius: 3px; }
    .stat.rec { border-color: #c00; color: #f55; }
    #controls { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }
    button { background: #1a1a1a; color: #ccc; border: 1px solid #333; padding: 6px 14px;
             cursor: pointer; font: 13px monospace; border-radius: 3px; }
    button:hover { background: #252525; }
    button.on { border-color: #4a9; color: #4a9; }
    #targets { display: flex; gap: 6px; align-items: center; }
    #targets label { color: #666; }
    #targets button { padding: 4px 10px; }
    #targets button.active { border-color: #77c; color: #77c; }
    #dpad { display: flex; flex-direction: column; align-items: center; gap: 4px; }
    #dpad div { display: flex; gap: 4px; }
    #dpad button { width: 44px; height: 44px; font-size: 18px; padding: 0; }
  </style>
</head>
<body>
  <img src="/stream">
  <div id="status">
    <span class="stat" id="s-fps">FPS: &ndash;</span>
    <span class="stat" id="s-tracked">Cats: &ndash;</span>
    <span class="stat" id="s-rec">REC: OFF</span>
    <span class="stat" id="s-servo">Servo: &ndash;</span>
    <span class="stat" id="s-angles" style="display:none"></span>
  </div>
  <div id="controls">
    <button id="btn-rec"   onclick="cmd('toggle_record')">&#9210; Record</button>
    <button id="btn-debug" onclick="cmd('toggle_debug')">Debug</button>
    <button                onclick="cmd('servo_mode')">Servo Mode</button>
    <button                onclick="cmd('center')">&#8982; Center</button>
  </div>
  <div id="dpad">
    <div><button onclick="cmd('tilt_up')">▲</button></div>
    <div>
      <button onclick="cmd('pan_left')">◀</button>
      <button onclick="cmd('pan_right')">▶</button>
    </div>
    <div><button onclick="cmd('tilt_down')">▼</button></div>
  </div>
  <div id="targets">
    <label>Target:</label>
    <button class="active" data-id="0" onclick="target(0)">Any</button>
    <button data-id="1" onclick="target(1)">#1</button>
    <button data-id="2" onclick="target(2)">#2</button>
    <button data-id="3" onclick="target(3)">#3</button>
    <button data-id="4" onclick="target(4)">#4</button>
    <button data-id="5" onclick="target(5)">#5</button>
  </div>"""

_HTML_SCRIPT = """
  <script>
    let activeTarget = 0;
    const ws = new WebSocket('ws://' + location.host + '/ws');
    ws.onmessage = e => {
      const s = JSON.parse(e.data);
      document.getElementById('s-fps').textContent = 'FPS: ' + (s.fps ?? '–');
      document.getElementById('s-tracked').textContent = 'Cats: ' + (s.tracked ?? '–');
      const rec = document.getElementById('s-rec');
      rec.textContent = s.recording ? 'REC: ON' : 'REC: OFF';
      rec.classList.toggle('rec', !!s.recording);
      document.getElementById('btn-rec').classList.toggle('on', !!s.recording);
      document.getElementById('btn-debug').classList.toggle('on', !!s.debug);
      document.getElementById('s-servo').textContent = 'Servo: ' + (s.servo_mode ?? '–');
      const ang = document.getElementById('s-angles');
      if (s.pan != null) {
        ang.textContent = 'Pan ' + s.pan + '° / Tilt ' + s.tilt + '°';
        ang.style.display = '';
      } else { ang.style.display = 'none'; }
      if (s.target !== activeTarget) {
        activeTarget = s.target ?? 0;
        document.querySelectorAll('#targets button').forEach(b =>
          b.classList.toggle('active', +b.dataset.id === activeTarget));
      }
    };
    function cmd(name, extra) { ws.send(JSON.stringify(Object.assign({cmd: name}, extra))); }
    function target(id) { cmd('target', {id}); }
    const keyMap = {
      ArrowLeft: 'pan_left', ArrowRight: 'pan_right',
      ArrowUp:   'tilt_up',  ArrowDown:  'tilt_down',
    };
    document.addEventListener('keydown', e => {
      if (keyMap[e.key]) { e.preventDefault(); cmd(keyMap[e.key]); }
    });
  </script>"""

_HTML_FOOT = """
</body>
</html>"""


class StreamServer:
    """MJPEG video stream + WebSocket control panel, served over HTTP.

    - GET /       browser UI with live feed and control buttons
    - GET /stream MJPEG video stream
    - WS  /ws     bidirectional: receive commands, push status JSON
    """

    def __init__(self, port=5000):
        self.port = port
        self._frame_lock = threading.Lock()
        self._frame_bytes = None
        self._status_lock = threading.Lock()
        self._status = {}
        self._cmd_queue = queue.Queue()

        if not FLASK_AVAILABLE:
            raise RuntimeError("flask is required — pip install flask flask-sock")

        app = Flask(__name__)
        import logging
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

        if FLASK_SOCK_AVAILABLE:
            sock = Sock(app)

            @sock.route("/ws")
            def ws_handler(ws):
                while True:
                    msg = None
                    try:
                        msg = ws.receive(timeout=0.5)
                    except Exception:
                        pass
                    if msg:
                        try:
                            self._cmd_queue.put_nowait(json.loads(msg))
                        except Exception:
                            pass
                    with self._status_lock:
                        payload = json.dumps(self._status)
                    try:
                        ws.send(payload)
                    except Exception:
                        break

        app.add_url_rule("/", "index", self._index)
        app.add_url_rule("/stream", "stream", self._stream_route)

        t = threading.Thread(
            target=lambda: app.run(host="0.0.0.0", port=port, threaded=True),
            daemon=True,
        )
        t.start()
        ctrl = " + browser controls" if FLASK_SOCK_AVAILABLE else " (pip install flask-sock for controls)"
        print(f"[STREAM] Live at http://0.0.0.0:{port}/{ctrl}")

    def push(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self._frame_lock:
                self._frame_bytes = buf.tobytes()

    def update_status(self, status: dict):
        with self._status_lock:
            self._status = status

    def get_command(self):
        try:
            return self._cmd_queue.get_nowait()
        except queue.Empty:
            return None

    def _generate(self):
        while True:
            with self._frame_lock:
                data = self._frame_bytes
            if data is None:
                time.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
            time.sleep(0.05)

    def _stream_route(self):
        return Response(self._generate(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    def _index(self):
        script = _HTML_SCRIPT if FLASK_SOCK_AVAILABLE else ""
        return _HTML_HEAD + script + _HTML_FOOT
