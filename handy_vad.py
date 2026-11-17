"""
handy_vad.py — fala -> Handy transcreve -> cola, sem apertar atalho e sem cortar o começo.

Fluxo:
  1. Silero VAD (modelo do próprio Handy) escuta o microfone o tempo todo,
     guardando sempre os últimos ~600 ms de áudio (pré-roll).
  2. Detectou fala -> aperta o atalho do Handy (Ctrl+Espaço): a barrinha dele aparece.
     Ao mesmo tempo o script grava a frase INTEIRA, incluindo o pré-roll.
  3. Silêncio -> Esc cancela a gravação do Handy (que começou atrasada e perderia
     a 1ª palavra) e a frase completa é transcrita pelo Handy com
     `handy.exe -f frase.wav` (mesmo modelo do app) e colada com Ctrl+V.

O Handy precisa estar aberto. Escuta no mesmo microfone configurado no Handy.

Requisitos: pip install sounddevice numpy onnxruntime keyboard pyperclip
Uso:        python handy_vad.py            (Ctrl+C para sair)
            python handy_vad.py --help     (ver ajustes)
"""

import argparse
import collections
import ctypes
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import wave
from ctypes import wintypes

import keyboard
import numpy as np
import onnxruntime as ort
import pyperclip
import sounddevice as sd

HANDY_DIR = r"C:\Users\PC\AppData\Local\Handy"
HANDY_EXE = os.path.join(HANDY_DIR, "handy.exe")
SILERO_MODEL = os.path.join(HANDY_DIR, "resources", "models", "silero_vad_v4.onnx")
HANDY_SETTINGS = os.path.expandvars(r"%APPDATA%\com.pais.handy\settings_store.json")
SAMPLE_RATE = 16000
FRAME_SAMPLES = 512                               # bloco exigido pelo Silero a 16 kHz
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE    # 32 ms
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
DEBUG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug")


def parse_args():
    p = argparse.ArgumentParser(description="Silero VAD -> Handy -> colar")
    p.add_argument("--handy", default=HANDY_EXE, help="caminho do handy.exe")
    p.add_argument("--vad-model", default=SILERO_MODEL, help="modelo Silero VAD (.onnx)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="probabilidade de voz p/ considerar fala (0-1). Padrão 0.5 "
                        "(maior = menos falso positivo, menor = pega fala mais baixa)")
    p.add_argument("--start-ms", type=int, default=64,
                   help="voz contínua necessária para iniciar (ms). Padrão 64")
    p.add_argument("--debug", action="store_true",
                   help="mostra quase-disparos e salva o áudio de cada frase em .\\debug")
    p.add_argument("--preroll-ms", type=int, default=600,
                   help="áudio de antes da detecção incluído na frase (ms). Padrão 600")
    p.add_argument("--silence-ms", type=int, default=1000,
                   help="silêncio contínuo para encerrar a fala (ms). Padrão 1000")
    p.add_argument("--min-speech-ms", type=int, default=0,
                   help="descarta falas com menos voz que isso (ms). Padrão 0 = nunca descarta; "
                        "o Handy decide (ruído vira 'nada reconhecido' e não cola nada)")
    p.add_argument("--max-sec", type=float, default=60.0,
                   help="duração máxima de uma frase (s). Padrão 60")
    p.add_argument("--bar", choices=("always", "speech", "off"), default="always",
                   help="barrinha do Handy: always = sempre aberta (padrão), "
                        "speech = só durante fala/transcrição, off = não mostrar")
    p.add_argument("--bar-refresh-sec", type=float, default=120.0,
                   help="no modo always, renova a gravação do Handy a cada N s "
                        "(em silêncio) p/ não acumular memória. Padrão 120")
    p.add_argument("--no-paste", action="store_true", help="só mostra o texto, não cola")
    p.add_argument("--no-meter", dest="meter", action="store_false",
                   help="não mostrar o medidor ao vivo")
    p.add_argument("--device", default=None,
                   help="índice ou nome do microfone (padrão: o mesmo do Handy; ver --list-devices)")
    p.add_argument("--list-devices", action="store_true", help="lista dispositivos e sai")
    p.add_argument("--calibrate", action="store_true",
                   help="só mostra a probabilidade de voz ao vivo")
    return p.parse_args()


def log(msg):
    print(f"\r{' ' * 80}\r[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def handy_settings():
    try:
        with open(HANDY_SETTINGS, encoding="utf-8") as f:
            return json.load(f)["settings"]
    except Exception:
        return {}


# ---------------------------------------------------------------- VAD

class SileroVAD:
    """Silero VAD v4 (ONNX). Recebe blocos int16 de 512 amostras, devolve prob. de voz."""

    def __init__(self, path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.h = np.zeros((2, 1, 64), dtype=np.float32)
        self.c = np.zeros((2, 1, 64), dtype=np.float32)

    def __call__(self, pcm16):
        x = (np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0)[None, :]
        out, self.h, self.c = self.sess.run(
            None, {"input": x, "sr": self.sr, "h": self.h, "c": self.c})
        return float(out[0][0])


# ---------------------------------------------------------------- Handy

_user32 = ctypes.windll.user32
_kernel32 = ctypes.windll.kernel32


def _exe_of(hwnd):
    pid = wintypes.DWORD()
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    h = _kernel32.OpenProcess(0x1000, False, pid.value)  # QUERY_LIMITED_INFORMATION
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(520)
        size = wintypes.DWORD(520)
        _kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size))
        return buf.value.lower()
    finally:
        _kernel32.CloseHandle(h)


def handy_bar_visible():
    """True se a barrinha de gravação do Handy ('Recording') está na tela."""
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if _user32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(32)
            _user32.GetWindowTextW(hwnd, buf, 32)
            if buf.value == "Recording" and _exe_of(hwnd).endswith("handy.exe"):
                found.append(hwnd)
                return False
        return True

    _user32.EnumWindows(cb, 0)
    return bool(found)


class HandyBar:
    """Controla a barrinha do Handy. Ela só aparece com o Handy gravando, então o
    script liga a gravação dele (Ctrl+Espaço) e descarta com Esc — quem transcreve
    é o `handy.exe -f` com o áudio completo do script.

    modos:  always = aberta o tempo todo (renova a gravação do Handy de tempos em
                     tempos, em silêncio, para não acumular áudio na memória)
            speech = abre ao detectar fala, fecha depois de colar
            off    = não mexe no Handy
    """

    def __init__(self, mode, refresh_sec):
        b = handy_settings().get("bindings", {})
        self.start_key = b.get("transcribe", {}).get("current_binding") or "ctrl+space"
        self.cancel_key = b.get("cancel", {}).get("current_binding") or "escape"
        self.mode = mode
        self.refresh_sec = refresh_sec
        self.active = False
        self.since = 0.0
        self.speaking = False
        self.pending = 0                  # frases na fila/em transcrição
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.thread = None
        if mode != "off":
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    # --- chamados pela escuta / worker
    def speech_start(self):
        with self.lock:
            self.speaking = True
        self.wake.set()

    def speech_end(self, queued):
        with self.lock:
            self.speaking = False
            if queued:
                self.pending += 1
        self.wake.set()

    def job_done(self):
        with self.lock:
            self.pending -= 1
        self.wake.set()

    def close(self):
        if self.thread:
            self.mode = "off"
            self.wake.set()
            self.thread.join(timeout=2)

    # --- controle do Handy
    def _start(self):
        # se o Handy já está gravando, NÃO aperta o atalho: seria um toggle que
        # pararia a gravação e faria o Handy transcrever/colar por conta própria
        if not handy_bar_visible():
            keyboard.send(self.start_key)
        self.active, self.since = True, time.monotonic()

    def _cancel(self):
        self.active = False
        # só manda Esc se o Handy está mesmo gravando (senão iria p/ a janela ativa)
        for _ in range(10):
            if handy_bar_visible():
                keyboard.send(self.cancel_key)
                break
            time.sleep(0.05)
        for _ in range(30):              # espera a barrinha fechar de fato
            if not handy_bar_visible():
                return
            time.sleep(0.05)

    def _loop(self):
        last_check = 0.0
        while True:
            self.wake.wait(0.25)
            self.wake.clear()
            with self.lock:
                busy = self.speaking or self.pending > 0
            now = time.monotonic()
            mode = self.mode
            want = mode == "always" or (mode == "speech" and busy)

            if want and not self.active:
                self._start()
            elif not want and self.active:
                self._cancel()
            elif self.active and mode == "always":
                if now - self.since > 2.0 and now - last_check > 1.0:
                    last_check = now
                    if not handy_bar_visible():          # o Handy fechou sozinho
                        log("[barrinha] Handy fechou a barrinha; reabrindo")
                        self.active = False
                        continue
                if not busy and now - self.since > self.refresh_sec:
                    self._cancel()                       # renova (em silêncio)
                    time.sleep(0.15)
            if mode == "off" and not self.active:
                return


def write_wav(path, pcm_bytes):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm_bytes)


def transcribe(args, pcm_bytes):
    """Transcreve a frase com o modelo do Handy: `handy.exe -f arquivo.wav --json`."""
    fd, path = tempfile.mkstemp(prefix="handyvad-", suffix=".wav")
    os.close(fd)
    try:
        write_wav(path, pcm_bytes)
        r = subprocess.run([args.handy, "-f", path, "--json"], capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           creationflags=NO_WINDOW)
        for line in reversed(r.stdout.splitlines()):
            if line.strip().startswith("{"):
                return json.loads(line).get("text", "").strip()
        log(f"[erro] Handy não retornou texto (código {r.returncode})")
        return ""
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def paste(text):
    """Cola via Ctrl+V e restaura a área de transferência anterior."""
    try:
        old = pyperclip.paste()
    except Exception:
        old = None
    pyperclip.copy(text)
    time.sleep(0.05)
    keyboard.send("ctrl+v")
    time.sleep(0.15)
    if old is not None:
        try:
            pyperclip.copy(old)
        except Exception:
            pass


def worker(args, jobs, suffix, bar):
    """Transcreve e cola as frases em ordem, sem travar a escuta."""
    while True:
        pcm = jobs.get()
        t0 = time.monotonic()
        try:
            if args.debug:
                os.makedirs(DEBUG_DIR, exist_ok=True)
                dbg = os.path.join(DEBUG_DIR, time.strftime("%H%M%S") + ".wav")
                write_wav(dbg, pcm)
                log(f"[debug] áudio salvo: {dbg}")
            text = transcribe(args, pcm)
            if not text:
                log("(nada reconhecido)")
            else:
                log(f"Texto ({time.monotonic() - t0:.1f}s): {text}")
                if not args.no_paste:
                    paste(text + suffix)
        except Exception as e:
            log(f"[erro] transcrição: {e}")
        finally:
            bar.job_done()   # barrinha do Handy some só depois de colar


# ---------------------------------------------------------------- escuta

def pick_device(arg):
    """--device > microfone configurado no Handy > padrão do Windows."""
    if arg is not None:
        return int(arg) if arg.isdigit() else arg
    name = handy_settings().get("selected_microphone")
    if name:
        for i, d in enumerate(sd.query_devices()):
            # MME corta nomes em 31 caracteres; compara pelo começo
            if d["max_input_channels"] > 0 and name.startswith(d["name"][:31]):
                return i
        log(f"[aviso] microfone do Handy '{name}' não encontrado; usando o padrão do Windows")
    for api in sd.query_hostapis():   # Mapeador de som (segue o padrão do Windows)
        if api["name"] == "MME" and api["devices"]:
            return api["devices"][0]
    return None


def main():
    args = parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return
    for path, what in ((args.handy, "handy.exe"), (args.vad_model, "modelo Silero")):
        if not os.path.exists(path):
            sys.exit(f"[erro] {what} não encontrado em: {path}")

    # uma cópia só: duas cópias apertando o atalho do Handy se atrapalham
    if not args.calibrate:
        _kernel32.CreateMutexW(None, False, "Local\\handy_vad_single_instance")
        if _kernel32.GetLastError() == 183:   # ERROR_ALREADY_EXISTS
            sys.exit("[erro] o handy_vad.py já está rodando em outro terminal.")

    device = pick_device(args.device)
    log(f"Microfone: {sd.query_devices(device)['name']}")

    vad = SileroVAD(args.vad_model)
    bar = HandyBar("off" if args.calibrate else args.bar, args.bar_refresh_sec)
    suffix = " " if handy_settings().get("append_trailing_space") else ""
    jobs = queue.Queue()
    if not args.calibrate:
        threading.Thread(target=worker, args=(args, jobs, suffix, bar), daemon=True).start()

    start_frames = max(1, args.start_ms // FRAME_MS)
    preroll_frames = max(start_frames, args.preroll_ms // FRAME_MS)
    silence_frames = max(1, args.silence_ms // FRAME_MS)
    min_voiced = args.min_speech_ms // FRAME_MS
    max_frames = int(args.max_sec * 1000 / FRAME_MS)
    keep_tail = min(silence_frames, 10)            # ~320 ms de silêncio no fim
    neg_threshold = max(0.05, args.threshold - 0.15)   # histerese p/ o fim da fala
    warmup_frames = 500 // FRAME_MS

    preroll = collections.deque(maxlen=preroll_frames)
    recording = False
    utter = []
    run = voiced = silent_run = 0
    frame_count = 0
    near_peak, near_len = 0.0, 0

    stream = sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME_SAMPLES,
                               channels=1, dtype="int16", device=device)
    try:
        with stream:
            log("Escutando... fale normalmente (Ctrl+C para sair)")
            while True:
                data, _ = stream.read(FRAME_SAMPLES)
                frame = bytes(data)
                prob = vad(frame)
                frame_count += 1
                if frame_count < warmup_frames:
                    continue

                if (args.calibrate or (args.meter and not recording)) and frame_count % 3 == 0:
                    meter = "#" * int(prob * 40)
                    print(f"\rvoz {prob:4.2f} {'FALA' if prob >= args.threshold else '    '} "
                          f"|{meter:<40}|", end="", flush=True)
                if args.calibrate:
                    continue

                if not recording:
                    preroll.append(frame)
                    if args.debug:   # registra "quase disparos" (som que parecia voz)
                        if prob >= 0.2:
                            near_peak, near_len = max(near_peak, prob), near_len + 1
                        elif near_len:
                            log(f"[debug] quase-fala ignorada: pico {near_peak:.2f}, "
                                f"{near_len * FRAME_MS} ms (precisa ≥{args.threshold} "
                                f"por {start_frames * FRAME_MS} ms seguidos)")
                            near_peak, near_len = 0.0, 0
                    run = run + 1 if prob >= args.threshold else 0
                    if run >= start_frames:
                        recording = True
                        utter = list(preroll)          # inclui o começo da fala
                        preroll.clear()
                        voiced, silent_run = run, 0
                        near_peak, near_len = 0.0, 0
                        bar.speech_start()
                        log("Fala detectada -> gravando")
                    continue

                utter.append(frame)
                if prob >= args.threshold:
                    voiced += 1
                silent_run = 0 if prob >= neg_threshold else silent_run + 1

                if silent_run >= silence_frames or len(utter) >= max_frames:
                    cut = max(0, silent_run - keep_tail)   # tira silêncio sobrando no fim
                    frames = utter[:len(utter) - cut] if cut else utter
                    recording, utter, run = False, [], 0
                    queued = voiced >= min_voiced
                    if queued:
                        log(f"Fim da fala ({len(frames) * FRAME_MS / 1000:.1f}s) -> transcrevendo...")
                        jobs.put(b"".join(frames))
                    else:
                        log(f"(descartado: só {voiced * FRAME_MS} ms de voz)")
                    bar.speech_end(queued)

    except KeyboardInterrupt:
        bar.close()
        print()
        log("Encerrado.")


if __name__ == "__main__":
    main()
